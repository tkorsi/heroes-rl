"""Synthetic capability/admission tamper controls; never mint native evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.learner import contextual as c, contextual_model as m, selfplay_training as sp
from experiments.learner.contextual_rollout import ContextualCollector
from experiments.learner.interaction_guard import REVEALED_VERSION
from experiments.learner.ppo import PPOConfig, PPOReport
from tests.test_contextual_rollout import initial, request, resolve, finish_pair


def paired(model, opponent, episode, seat, *, purpose='training'):
    collector = ContextualCollector((model, opponent) if seat == 0 else (opponent, model),
        ('learner', 'opponent') if seat == 0 else ('opponent', 'learner'), learner_player=seat,
        opponent_pool=sp.POOL, interface=c.EXPANDED_ENVIRONMENT, purpose=purpose, origin='synthetic',
        policy_seeds=(11, 12), guard_version=REVEALED_VERSION)
    collector.start(str(episode))
    pair = tuple(initial(p, str(episode), schema=c.EXPANDED_SCHEMA) for p in (0, 1))
    for h in pair:
        resolve(h, collector.act(request(h)))
    end = finish_pair(pair, 'natural'); end.manifest['dialog_cutoff_drain'] = None
    result = collector.finish(end, histories=tuple(h.frames for h in pair))
    return replace(result, rollouts=tuple(replace(r, origin='native') for r in result.rollouts))


class SelfplayTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.manual_seed(401)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.opponent = deepcopy(self.model).eval().requires_grad_(False)
        self.config = PPOConfig()
        self.plan = dict(mode='training', execution_output=str(self.root / 'execution'), configuration=dict(
            ppo=self.config.to_dict(), adam={'lr': 0.0003, 'betas': [0.9, 0.999]}, max_optimizer_steps=640),
            schedule=[dict(episode=i+1, learner_player=i%2, policy_seeds=[11,12]) for i in range(4)])
        self.digest = 'a' * 64
        # Tests deliberately inject synthetic receipts into a private registry. Production
        # can populate it only after fresh owned-child completion and actual native replay.
        for name in ('_BINDINGS', '_LAUNCHES', '_COMPLETIONS', '_RECEIPTS'):
            context = patch.object(sp, name, {}); context.start(); self.addCleanup(context.stop)
        context = patch.object(sp, '_USED', set()); context.start(); self.addCleanup(context.stop)
        context = patch.object(sp, '_plan', return_value=self.plan); context.start(); self.addCleanup(context.stop)
        self.receipts = []
        for index in range(4):
            folder = self.root / 'execution' / f'episode-{index+1:06d}'; folder.mkdir(parents=True)
            bound = sp.CaptureBinding(str(self.root / 'plan.json'), self.digest, index, str(folder), sp.canonical(self.plan))
            sp._BINDINGS[id(bound)] = bound
            collected = paired(self.model, self.opponent, index+1, index%2)
            sp.write_new(folder / 'neural-rollout.json', collected.to_dict())
            files = {'neural-rollout.json': sp.evidence.file_identity(folder / 'neural-rollout.json')}
            receipt = sp.VerifiedTrainingReceipt(bound, collected, sp.canonical(files), sp.canonical({'synthetic_fixture': True}))
            sp.write_new(folder / 'training-receipt.json', receipt.to_dict())
            sp._RECEIPTS[id(receipt)] = receipt
            self.receipts.append(receipt)
        self.receipts = tuple(self.receipts)

    def test_exact_two_private_policy_reconstruction_and_wrong_opponent_rejects(self):
        for receipt in self.receipts[:2]:
            models = (self.model, self.opponent) if receipt.rollout.player == 0 else (self.opponent, self.model)
            proof = sp.reconstruct_policies(receipt.collected, models, [11,12], lambda: 10)
            self.assertEqual([p['player'] for p in proof], [0,1])
            self.assertTrue(all(p['private_memory_reset'] and p['maximum_absolute_difference'] == 0 for p in proof))
        changed = deepcopy(self.opponent)
        with torch.no_grad(): next(changed.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'weights'):
            sp.reconstruct_policies(self.receipts[0].collected, (self.model, changed), [11,12], lambda: 10)
        with self.assertRaises(TimeoutError):
            sp.reconstruct_policies(self.receipts[0].collected, (self.model, self.opponent), [11,12], lambda: 0)

    def test_real_controller_and_verifier_dependency_inventories_match(self):
        from tools import run_selfplay_training as controller
        with patch('tools.benchmark_copied_training.checkout_identity', return_value={'fixture': True}):
            self.assertEqual(controller.source_identity()['implementation'], sp.implementation_identity())

    def test_only_designated_learner_seats_enter_unchanged_ppo_once(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        expected = tuple(r.collected.rollouts[i%2] for i, r in enumerate(self.receipts))
        with patch('experiments.learner.builtin_gameplay.transactional_update', return_value=PPOReport(4,2,0,0,0,0)) as update:
            result, rollouts = sp.native_update(self.model, optimizer, self.receipts, plan_sha256=self.digest,
                config=self.config, generator=torch.Generator(), remaining=lambda: 10)
        self.assertEqual(rollouts, expected)
        self.assertEqual([r.player for r in rollouts], [0,1,0,1])
        self.assertTrue(all(r.role == 'learner' for r in rollouts))
        self.assertEqual(update.call_args.args[2], expected)
        self.assertIs(update.call_args.kwargs['config'], self.config)
        self.assertEqual(result.purpose, sp.PROVENANCE)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256=self.digest)

    def test_partial_duplicate_reordered_forged_and_wrong_plan_never_admit(self):
        invalid = (self.receipts[:3], (*self.receipts[:3], self.receipts[0]), self.receipts[::-1],
                   (*self.receipts[:3], replace(self.receipts[3])))
        for receipts in invalid:
            with self.subTest(receipts=len(receipts)), self.assertRaises(ValueError):
                sp._admit_selfplay(receipts, self.model, plan_sha256=self.digest)
        with self.assertRaisesRegex(ValueError, 'plan/purpose'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256='b'*64)

    def test_diagnostic_receipts_are_never_admitted(self):
        self.plan['mode'] = 'diagnostic'
        for receipt in self.receipts:
            object.__setattr__(receipt.binding, 'plan_json', sp.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, 'plan/purpose'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256=self.digest)

    def test_changed_evidence_or_weights_rejects_before_update(self):
        path = Path(self.receipts[0].binding.folder) / 'neural-rollout.json'; old = path.read_bytes()
        path.write_bytes(old+b' ')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256=self.digest)
        path.write_bytes(old)
        with torch.no_grad(): next(self.model.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'off-policy'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256=self.digest)

    def test_failed_update_consumes_capabilities_and_cannot_be_retried(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        with patch('experiments.learner.builtin_gameplay.transactional_update', side_effect=TimeoutError('injected')):
            with self.assertRaises(TimeoutError):
                sp.native_update(self.model, optimizer, self.receipts, plan_sha256=self.digest,
                    config=self.config, generator=torch.Generator(), remaining=lambda: 10)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            sp._admit_selfplay(self.receipts, self.model, plan_sha256=self.digest)

    def test_completion_requires_registered_successful_child_private_token_and_unchanged_files(self):
        bound = self.receipts[0].binding
        folder = Path(bound.folder); sp.write_new(folder / 'manifest.json', {'fixture': True})
        message = dict(schema=sp.COMPLETION_SCHEMA, token='a'*64, episode=1,
            files={n:sp.evidence.file_identity(folder/n) for n in ('manifest.json','neural-rollout.json')})
        with self.assertRaises(ValueError): sp.accept_capture_completion(bound, message)
        process = SimpleNamespace(pid=123, poll=lambda: None)
        sp.register_capture_launch(bound, process, 'a'*64)
        with self.assertRaises(ValueError): sp.accept_capture_completion(bound, message)
        process.poll = lambda: 1
        with self.assertRaises(ValueError): sp.accept_capture_completion(bound, message)
        process.poll = lambda: 0
        with self.assertRaises(ValueError): sp.accept_capture_completion(bound, dict(message, token='b'*64))
        sp.accept_capture_completion(bound, message)
        with self.assertRaises(ValueError): sp.accept_capture_completion(bound, message)

    def test_replay_verification_cannot_start_from_a_disk_receipt(self):
        bound = self.receipts[0].binding
        with self.assertRaisesRegex(ValueError, 'freshly completed'):
            sp.verify_training_capture(bound, self.model, self.opponent, replay_binary='/absent',
                                       data='/absent', remaining=lambda: 10)

    def test_native_replay_rejection_cannot_mint_receipt_after_exact_policy_checks(self):
        receipt = self.receipts[0]; bound = receipt.binding; folder = Path(bound.folder)
        replay = self.root / 'synthetic-verifier'; replay.write_bytes(b'not an executable; native result is mocked')
        self.plan.update(replay_binary=sp.binding(replay), data=str(self.root))
        object.__setattr__(bound, 'plan_json', sp.canonical(self.plan))
        manifest = json.loads(receipt.collected.manifest_json)
        sp.write_new(folder / 'manifest.json', manifest)
        sp._COMPLETIONS[id(bound)] = sp.canonical({name:sp.evidence.file_identity(folder/name)
            for name in ('manifest.json','neural-rollout.json')})
        sp._RECEIPTS.pop(id(receipt))
        histories = {'verified': True, 'players': {'red': {}, 'blue': {}}}
        with patch.object(sp, '_check_collected', return_value=manifest), \
             patch.object(sp, '_files', return_value=json.loads(receipt.files_json)), \
             patch.object(sp.contextual_replay, 'bound_evidence', return_value={}), \
             patch('experiments.headless.contextual_replay_history.verify_histories', return_value=histories), \
             patch.object(sp.contextual_replay, 'verify_episode', return_value={'verified':False}) as native, \
             patch.object(sp, 'reconstruct_policies', wraps=sp.reconstruct_policies) as policies:
            with self.assertRaisesRegex(ValueError, 'native replay failed'):
                sp.verify_training_capture(bound, self.model, self.opponent, replay_binary=replay,
                                           data=self.root, remaining=lambda: 10)
        self.assertEqual(policies.call_count, 1)
        self.assertEqual(native.call_args.kwargs['binary'], replay)
        self.assertFalse(any(r.binding is bound for r in sp._RECEIPTS.values()))
        self.assertFalse((folder/'training-policy-reconstruction.json').exists())
