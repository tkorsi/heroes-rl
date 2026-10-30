"""Exact synthetic self-play checkpoint controls; no gameplay is admitted."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.learner import checkpoints as cp, contextual as c, contextual_model as m
from experiments.learner import selfplay_training as sp
from experiments.learner.ppo import PPOConfig
from tests.test_builtin_gameplay_checkpoint import equal_state


class SelfplayCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        torch.manual_seed(381); random.seed(382)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.pool = cp.FrozenOpponentPool(); self.pool.add('archived-not-current-opponent', self.model)
        self.generator = torch.Generator().manual_seed(383)
        self.initial = self.root / 'initial.pt'
        cp.save_policy_checkpoint(self.initial, self.model, checkpoint='fixture-initial', source_metadata={'fixture': True},
                                  interface=c.EXPANDED_ENVIRONMENT, opponent_pool=self.pool)
        self.meta = sp.metadata('a' * 64, sp.binding(self.initial), sp.binding(self.initial), campaign_id='fixture')
        self.ppo = PPOConfig().to_dict()
        self.progress = cp.TrainingProgress(3, 12, 57)
        self.sampler = {'next_sequence': 4, 'consumed_seeds': [101, 102]}
        # Populate Adam for every parameter with an explicitly synthetic scalar loss.
        sum(p.square().sum() for p in self.model.parameters()).backward(); self.optimizer.step()

    def snapshot(self):
        return deepcopy(dict(model=dict(self.model.state_dict()), optimizer=self.optimizer.state_dict(),
            torch=torch.get_rng_state(), python=random.getstate(), generator=self.generator.get_state(),
            pool=self.pool.state_dict(), sampler=self.sampler, progress=vars(self.progress)))

    def save(self, name='saved', **changes):
        values = dict(checkpoint='fixture-selfplay-3', ppo_config=self.ppo, progress=self.progress, metadata=self.meta,
                      opponent_pool=self.pool, sampler_state=self.sampler, generator=self.generator)
        values.update(changes)
        return sp.save_checkpoint_pair(self.root / name, self.model, self.optimizer, **values)

    def test_exact_complete_pair_and_caller_rng_are_preserved(self):
        before = self.snapshot(); receipt = self.save()
        equal_state(self, self.snapshot(), before)
        self.assertTrue(receipt['exact_model_adam_rng_progress_pool'])
        self.assertEqual(receipt['progress'], vars(self.progress))
        policy = sp.load_policy(sp.checked(receipt['policy']), self.meta)
        self.assertEqual(policy.provenance, cp.SELFPLAY_GAMEPLAY_PROVENANCE)
        self.assertEqual(policy.checkpoint, receipt['checkpoint'])
        equal_state(self, policy.model.state_dict(), self.model.state_dict())
        rng = torch.Generator(); restored = deepcopy(self.model)
        optimizer = torch.optim.Adam(restored.parameters(), lr=0.9)
        state = cp.load_checkpoint(sp.checked(receipt['resume']), restored, optimizer, expected_ppo_config=self.ppo,
            expected_source_metadata=self.meta, generators={'minibatch': rng}, at_rollout_boundary=True,
            interface=c.EXPANDED_ENVIRONMENT, allowed_provenance=(sp.PROVENANCE,))
        equal_state(self, optimizer.state_dict(), self.optimizer.state_dict())
        equal_state(self, rng.get_state(), self.generator.get_state())
        self.assertEqual(state.sampler_state, self.sampler)
        self.assertEqual(state.progress, self.progress)

    def test_rng_restored_even_when_roundtrip_fails(self):
        before = self.snapshot()
        def fail(*args, **kwargs):
            torch.rand(3); random.random(); self.generator.manual_seed(1)
            raise ValueError('injected verification failure')
        with patch.object(sp, 'load_policy', side_effect=fail), self.assertRaisesRegex(ValueError, 'injected'):
            self.save()
        equal_state(self, self.snapshot(), before)
        self.assertTrue((self.root / 'saved' / 'after-resume.pt').is_file())

    def test_metadata_requires_new_v8_contract_and_explicit_purpose(self):
        mutations = [lambda x: x.pop('purpose'), lambda x: x.update(purpose='approved'),
            lambda x: x.update(learner_only_ppo=False), lambda x: x.update(frozen_opponent=False),
            lambda x: x.update(native_opponent={}), lambda x: x.update(opponent_pool='archived'),
            lambda x: x.update(policy_support='visible-interaction-budget-v1'),
            lambda x: x['opponent_checkpoint'].update(bytes=True), lambda x: x.pop('initial_checkpoint')]
        for mutate in mutations:
            changed = deepcopy(self.meta); mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cp._check_gameplay_provenance(sp.PROVENANCE, changed, c.EXPANDED_ENVIRONMENT)
        with self.assertRaises(ValueError):
            cp._check_gameplay_provenance(sp.PROVENANCE, self.meta, c.ARMIES_ENVIRONMENT)
        with self.assertRaises(ValueError):
            cp._check_gameplay_provenance('approved_gameplay', {'plan_sha256': 'a'*64}, c.EXPANDED_ENVIRONMENT)
        cp._check_gameplay_provenance('approved_gameplay', {'plan_sha256': 'a'*64}, c.ARMIES_ENVIRONMENT)

    def test_diagnostic_purpose_is_preserved_and_cannot_be_relabeled_on_load(self):
        diagnostic = dict(self.meta, purpose='diagnostic')
        receipt = self.save(metadata=diagnostic)
        self.assertEqual(receipt['metadata']['purpose'], 'diagnostic')
        with self.assertRaises(ValueError):
            sp.load_policy(sp.checked(receipt['policy']), self.meta)
        with self.assertRaises(ValueError):
            cp.load_policy_checkpoint(sp.checked(receipt['policy']), interface=c.EXPANDED_ENVIRONMENT,
                                      allowed_provenance=(sp.PROVENANCE,))
        with self.assertRaises(ValueError):
            sp.load_policy(self.initial, {'fixture': True}, allowed_provenance=('random_initialization',))

    def test_corrupted_adam_dtype_is_rejected_before_restore(self):
        receipt = self.save()
        payload = torch.load(sp.checked(receipt['resume']), weights_only=False)
        state = payload['resume']['optimizer_state']['state']
        first = next(iter(state.values())); first['exp_avg'] = first['exp_avg'].double()
        tamper = self.root / 'tampered.pt'; torch.save(payload, tamper)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            cp.load_checkpoint(tamper, self.model, self.optimizer, expected_ppo_config=self.ppo,
                expected_source_metadata=self.meta, generators={'minibatch': self.generator}, at_rollout_boundary=True,
                interface=c.EXPANDED_ENVIRONMENT, allowed_provenance=(sp.PROVENANCE,))
        equal_state(self, self.snapshot(), before)

    def test_published_pair_restores_every_component_and_freezes_returned_policy(self):
        sampler = dict(self.sampler, optimizer_steps=1)
        receipt = self.save(sampler_state=sampler)
        before = self.snapshot()
        torch.manual_seed(999); random.seed(999)
        restored = sp.restore_checkpoint_pair(receipt, expected_metadata=self.meta,
            expected_ppo_config=self.ppo, expected_optimizer_steps=1)
        self.assertEqual(restored.progress, self.progress)
        self.assertEqual(restored.sampler_state, sampler)
        self.assertEqual(restored.optimizer_steps, 1)
        self.assertFalse(restored.model.training)
        self.assertTrue(all(not p.requires_grad for p in restored.model.parameters()))
        equal_state(self, dict(restored.model.state_dict()), dict(self.model.state_dict()))
        equal_state(self, restored.optimizer.state_dict(), before['optimizer'])
        equal_state(self, restored.pool.state_dict(), before['pool'])
        equal_state(self, restored.generator.get_state(), before['generator'])
        equal_state(self, torch.get_rng_state(), before['torch'])
        equal_state(self, random.getstate(), before['python'])
        self.assertTrue(restored.bindings['exact_model_adam_rng_progress_pool'])

    def test_restore_rejects_diagnostic_descriptor_drift_and_wrong_optimizer_count(self):
        receipt = self.save(sampler_state=dict(self.sampler, optimizer_steps=1))
        changes = (dict(receipt, checkpoint='unpublished'), dict(receipt, exact_model_adam_rng_progress_pool=False),
                   dict(receipt, progress={**receipt['progress'], 'updates': 4}),
                   dict(receipt, metadata={**self.meta, 'purpose': 'diagnostic'}))
        before = self.snapshot()
        for changed in changes:
            with self.subTest(changed=changed['checkpoint']), self.assertRaises(ValueError):
                sp.restore_checkpoint_pair(changed, expected_metadata=self.meta,
                    expected_ppo_config=self.ppo, expected_optimizer_steps=1)
            equal_state(self, self.snapshot(), before)
        with self.assertRaises(ValueError):
            sp.restore_checkpoint_pair(receipt, expected_metadata=self.meta,
                expected_ppo_config=self.ppo, expected_optimizer_steps=2)
        diagnostic = self.save('diagnostic', metadata={**self.meta, 'purpose': 'diagnostic'},
                               sampler_state=dict(self.sampler, optimizer_steps=1))
        with self.assertRaises(ValueError):
            sp.restore_checkpoint_pair(diagnostic, expected_metadata=diagnostic['metadata'],
                expected_ppo_config=self.ppo, expected_optimizer_steps=1)
        equal_state(self, self.snapshot(), before)

    def test_restore_rejects_matching_weights_with_different_policy_rng(self):
        receipt = self.save(sampler_state=dict(self.sampler, optimizer_steps=1))
        payload = torch.load(sp.checked(receipt['policy']), weights_only=True)
        payload['rng']['torch_cpu'] = torch.Generator().manual_seed(999).get_state()
        changed = self.root / 'wrong-policy-rng.pt'; torch.save(payload, changed)
        tampered = dict(receipt, policy=sp.binding(changed))
        before = self.snapshot()
        with self.assertRaises(ValueError):
            sp.restore_checkpoint_pair(tampered, expected_metadata=self.meta,
                expected_ppo_config=self.ppo, expected_optimizer_steps=1)
        equal_state(self, self.snapshot(), before)

    def test_restore_rejects_serialized_adam_cast_and_preserves_rng_on_late_failure(self):
        receipt = self.save(sampler_state=dict(self.sampler, optimizer_steps=1))
        payload = torch.load(sp.checked(receipt['resume']), weights_only=True)
        next(iter(payload['resume']['optimizer_state']['state'].values()))['exp_avg_sq'] = (
            next(iter(payload['resume']['optimizer_state']['state'].values()))['exp_avg_sq'].double())
        changed = self.root / 'wrong-adam.pt'; torch.save(payload, changed)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            sp.restore_checkpoint_pair(dict(receipt, resume=sp.binding(changed)), expected_metadata=self.meta,
                expected_ppo_config=self.ppo, expected_optimizer_steps=1)
        equal_state(self, self.snapshot(), before)
        actual = cp.load_checkpoint
        def fail_after_restore(*args, **kwargs):
            actual(*args, **kwargs)
            torch.rand(1); random.random()
            raise ValueError('injected post-restore failure')
        with patch.object(cp, 'load_checkpoint', side_effect=fail_after_restore), self.assertRaisesRegex(ValueError, 'injected'):
            sp.restore_checkpoint_pair(receipt, expected_metadata=self.meta,
                expected_ppo_config=self.ppo, expected_optimizer_steps=1)
        equal_state(self, self.snapshot(), before)
