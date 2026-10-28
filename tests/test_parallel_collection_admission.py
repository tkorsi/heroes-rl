"""Synthetic eight-game admission contracts; these are not native-game evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests import test_builtin_gameplay as fixtures
from experiments.learner import builtin_gameplay as gameplay
from experiments.learner import contextual_model as models
from experiments.learner.gameplay import assert_same_state
from experiments.learner.ppo import PPOConfig
from experiments.headless import process_deadline

torch = fixtures.torch


class ParallelBatchAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        context = torch.random.fork_rng(devices=[])
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.model = models.ContextualActorCritic(models.ModelConfig(
            schema=models.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.model.eval().requires_grad_(False)

    def batch(self, *, workers=8, enabled=True, rows=8, allow_prefix=False, cap=1280, production=False,
              contract=None, expanded=True):
        path, plan = fixtures.write_plan(self.root, rows=rows, allow_prefix=allow_prefix)
        plan['schema'] = 'heroes-builtin-training-v1-plan'
        cfg = plan['configuration']
        cfg['max_optimizer_steps'] = cap
        if expanded:
            cfg.update(ordinary_choice_limit=4960, reserved_dialog_choices=160,
                       max_total_choices=5120, max_days=365)
        if enabled:
            cfg['parallel_collection'] = ({'batch_size': 8, 'workers': workers}
                                          if contract is None else contract)
        if production:
            plan['execution'] = {'collection_workers': workers}
        path.write_text(json.dumps(plan))
        receipts = []
        for index, row in enumerate(plan['schedule']):
            bound = gameplay.TrainingBinding.create(
                plan_path=path, plan_sha256=gameplay.digest(path), schedule_index=index,
                folder=self.root / f"episode-{row['episode']:06d}", checkpoint='synthetic-fixed',
                checkpoint_file=plan['checkpoint']['path'], weights_sha256=models.weights_sha256(self.model),
                opponent_identity=fixtures.evidence.opponent_identity(row['learner_player'], 'a'*64),
                source_identity=plan['source'], native_identity=plan['native'],
                max_days=cfg['max_days'], nominal_choices=cfg['ordinary_choice_limit'],
                max_choices=cfg['max_total_choices'])
            folder = Path(bound.folder)
            folder.mkdir()
            experience = fixtures.synthetic_experience(self.model, player=row['learner_player'],
                                                       episode=row['episode'], seed=row['engine_seed'])
            reference = 'synthetic-only-external-proof'
            rollout = gameplay.BuiltinTrainingRollout(experience, bound, reference,
                                                     gameplay._samples(experience, reference))
            watched = folder / 'synthetic-evidence.json'
            watched.write_text('{}')
            receipt = gameplay.VerifiedTrainingReceipt(bound, rollout,
                json.dumps({watched.name: fixtures.evidence.native.file_identity(watched)}),
                str(self.root / 'synthetic-verifier'), '{}', '{}')
            (folder / 'training-receipt.json').write_text(json.dumps(receipt.to_dict()))
            # Explicit synthetic capability injection, as in the original admission tests.
            gameplay._RECEIPTS[id(receipt)] = receipt
            self.addCleanup(gameplay._RECEIPTS.pop, id(receipt), None)
            self.addCleanup(gameplay._USED_RECEIPTS.discard, id(receipt))
            self.addCleanup(gameplay._BINDINGS.pop, id(bound), None)
            receipts.append(receipt)
        return path, plan, tuple(receipts)

    def admit(self, path, receipts):
        with patch('experiments.headless.builtin_replay.verify_identity'):
            return gameplay._admit_builtin(receipts, self.model, plan_sha256=gameplay.digest(path))

    def test_eight_requires_explicit_contract_even_when_all_receipts_are_fresh(self):
        path, _, receipts = self.batch(enabled=False)
        with self.assertRaises(ValueError):
            self.admit(path, receipts)

    def test_explicit_eight_preserves_planned_order_and_parent_model_rng(self):
        path, _, receipts = self.batch()
        state = deepcopy(self.model.state_dict())
        torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        admitted = self.admit(path, receipts)
        self.assertEqual(admitted, tuple(receipt.rollout for receipt in receipts))
        assert_same_state(self.model.state_dict(), state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(random.getstate(), python_rng)
        for bad in (receipts[::-1], receipts[:4], receipts[:-1], receipts[1:],
                    (receipts[0],)*8, receipts + receipts[:1]):
            with self.subTest(indices=[r.binding.schedule_index for r in bad]), self.assertRaises(ValueError):
                self.admit(path, bad)

    def test_serial_comparator_admits_same_complete_eight_game_contract(self):
        path, _, receipts = self.batch(workers=1)
        self.assertEqual(len(self.admit(path, receipts)), 8)

    def test_production_four_keeps_complete_batch_and_original_optimizer_cap(self):
        path, plan, receipts = self.batch(enabled=False, rows=4, cap=640, production=True, workers=4)
        self.assertNotIn('parallel_collection', plan['configuration'])
        self.assertEqual(len(self.admit(path, receipts)), 4)
        with self.assertRaisesRegex(ValueError, 'complete explicit batch'):
            self.admit(path, receipts[:3])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        with patch('experiments.headless.builtin_replay.verify_identity'):
            result, admitted = gameplay.native_update(self.model, optimizer, receipts,
                plan_sha256=gameplay.digest(path), config=PPOConfig(epochs=1),
                generator=torch.Generator().manual_seed(461), remaining=lambda: 60.,
                max_optimizer_steps=640)
        self.assertEqual(len(admitted), 4)
        self.assertEqual((result.samples, result.optimizer_steps), (8, 2))
        with self.assertRaisesRegex(ValueError, 'already used'):
            self.admit(path, receipts)

    def test_four_worker_execution_never_allows_expanded_eight_game_cap(self):
        path, _, receipts = self.batch(enabled=False, rows=4, cap=1280, production=True, workers=4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        with patch('experiments.headless.builtin_replay.verify_identity'), \
                patch.object(gameplay, 'transactional_update') as update, self.assertRaises(ValueError):
            gameplay.native_update(self.model, optimizer, receipts,
                plan_sha256=gameplay.digest(path), config=PPOConfig(epochs=1),
                generator=torch.Generator(), remaining=lambda: 60., max_optimizer_steps=1280)
        update.assert_not_called()

    def test_explicit_serial_four_comparator_uses_same_complete_contract(self):
        path, _, receipts = self.batch(enabled=False, rows=4, cap=640, production=True, workers=1)
        self.assertEqual(len(self.admit(path, receipts)), 4)

    def test_eight_rejects_legacy_prefix_permission(self):
        # Collection limits are small solely so TrainingBinding's existing
        # expanded-limit guard does not preempt the eight-game batch guard.
        path, _, receipts = self.batch(allow_prefix=True, cap=32, expanded=False)
        with self.assertRaises(ValueError):
            self.admit(path, receipts)
        with self.assertRaises(ValueError):
            self.admit(path, receipts[:4])

    def test_malformed_or_non_eight_contract_never_expands_admission(self):
        # Each case gets its own immutable plan and fresh receipt identities.
        contracts = ({'batch_size': 8, 'workers': 2}, {'batch_size': 9, 'workers': 8},
                     {'batch_size': True, 'workers': 8}, {'batch_size': 8, 'workers': True},
                     {'batch_size': 8, 'workers': 8, 'allow_retry': True},
                     {'batch_size': 8}, {})
        original = self.root
        for index, contract in enumerate(contracts):
            with self.subTest(contract=contract):
                self.root = original / str(index)
                self.root.mkdir()
                path, _, receipts = self.batch(contract=contract)
                with self.assertRaises(ValueError):
                    self.admit(path, receipts)
        self.root = original

    def test_eight_rejects_cross_plan_receipt_and_changed_checkpoint(self):
        path, _, receipts = self.batch()
        with patch('experiments.headless.builtin_replay.verify_identity'), self.assertRaises(ValueError):
            gameplay._admit_builtin(receipts, self.model, plan_sha256='f'*64)
        Path(json.loads(receipts[0].binding.checkpoint_file_json)['path']).write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.admit(path, receipts)

    def test_actual_tiny_eight_game_update_consumes_all_synthetic_samples_once(self):
        path, _, receipts = self.batch()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        initial = models.weights_sha256(self.model)
        with patch('experiments.headless.builtin_replay.verify_identity'):
            result, admitted = gameplay.native_update(self.model, optimizer, receipts,
                plan_sha256=gameplay.digest(path), config=PPOConfig(epochs=1),
                generator=torch.Generator().manual_seed(461), remaining=lambda: 60.,
                max_optimizer_steps=1280)
        self.assertEqual(len(admitted), 8)
        self.assertEqual(result.samples, 16)
        self.assertEqual(result.optimizer_steps, 4)
        self.assertNotEqual(models.weights_sha256(self.model), initial)
        with self.assertRaisesRegex(ValueError, 'already used'):
            self.admit(path, receipts)

    def test_legacy_four_keeps_its_original_optimizer_cap(self):
        path, _, receipts = self.batch(enabled=False, rows=4, cap=1280)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        with patch('experiments.headless.builtin_replay.verify_identity'), \
                patch.object(gameplay, 'transactional_update') as update, self.assertRaises(ValueError):
            gameplay.native_update(self.model, optimizer, receipts,
                plan_sha256=gameplay.digest(path), config=PPOConfig(epochs=1),
                generator=torch.Generator(), remaining=lambda: 60., max_optimizer_steps=1280)
        update.assert_not_called()


@unittest.skipUnless(os.name == 'posix' and hasattr(os, 'fork'), 'POSIX guarded collection')
class LiveCaptureTransferTests(unittest.TestCase):
    """Real private pipes transfer synthetic typed records, not native receipts."""
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        context = torch.random.fork_rng(devices=[])
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.model = models.ContextualActorCritic(models.ModelConfig(
            schema=models.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.path, self.plan = fixtures.write_plan(self.root, rows=2)
        self.binding = fixtures.binding_for(self.root, self.model, self.path, self.plan, 0)
        self.addCleanup(gameplay._BINDINGS.pop, id(self.binding), None)

    def lease(self):
        lease = gameplay._begin_parallel_capture(self.binding)
        self.addCleanup(gameplay._PARALLEL_CAPTURES.pop, id(lease), None)
        return lease

    def child(self, lease, *, message_changes=None, capture_changes=None):
        experience = fixtures.synthetic_experience(self.model)
        capture = gameplay.TrainingCapture(experience, self.binding).to_dict()
        capture.update(capture_changes or {})
        payload = dict(folder=self.binding.folder, capture=capture, token=lease.token,
                       schedule_index=self.binding.schedule_index,
                       plan_sha256=self.binding.plan_sha256, message_changes=message_changes or {})
        program = '''
import hashlib, json, pathlib, sys
item = json.load(sys.stdin)
folder = pathlib.Path(item['folder'])
folder.mkdir()
path = folder / 'builtin-training-rollout.json'
path.write_text(json.dumps(item['capture']))
data = path.read_bytes()
message = dict(token=item['token'], schedule_index=item['schedule_index'],
               plan_sha256=item['plan_sha256'], status='passed',
               capture=dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
message.update(item['message_changes'])
print(json.dumps(message), flush=True)
'''
        process = process_deadline.guarded_popen([sys.executable, '-c', program],
            deadline_monotonic=time.monotonic()+10, deadline_wall=time.time()+10,
            evidence_path=self.root/'guard.jsonl', stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(process.stderr.close)
        self.addCleanup(process.stdout.close)
        self.addCleanup(process_deadline.close_owned_process, process)
        gameplay._attach_parallel_process(lease, process)
        process.stdin.write(json.dumps(payload).encode())
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5), 0, process.stderr.read().decode())
        return process

    def test_live_child_transfer_is_one_shot_and_does_not_mint_a_receipt(self):
        lease = self.lease()
        process = self.child(lease)
        existing = set(gameplay._RECEIPTS)
        state = deepcopy(self.model.state_dict())
        torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        capture, message = gameplay._complete_parallel_capture(lease)
        self.addCleanup(gameplay._CAPTURES.pop, id(capture), None)
        self.assertIs(capture.binding, self.binding)
        self.assertIs(gameplay._CAPTURES[id(capture)], capture)
        self.assertFalse(capture.experience.optimization_eligible)
        self.assertEqual(set(gameplay._RECEIPTS), existing)
        self.assertEqual(message['token'], lease.token)
        self.assertEqual(process.returncode, 0)
        assert_same_state(self.model.state_dict(), state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(random.getstate(), python_rng)
        with self.assertRaisesRegex(ValueError, 'fresh live'):
            gameplay._complete_parallel_capture(lease)

    def test_result_file_without_attached_live_process_cannot_import(self):
        lease = self.lease()
        folder = Path(self.binding.folder)
        folder.mkdir()
        capture = gameplay.TrainingCapture(fixtures.synthetic_experience(self.model), self.binding)
        (folder/'builtin-training-rollout.json').write_text(json.dumps(capture.to_dict()))
        with self.assertRaisesRegex(ValueError, 'fresh live'):
            gameplay._complete_parallel_capture(lease)
        self.assertNotIn(id(capture), gameplay._CAPTURES)

    def test_copied_or_cancelled_lease_cannot_import(self):
        from copy import copy
        lease = self.lease()
        self.child(lease)
        with self.assertRaisesRegex(ValueError, 'fresh live'):
            gameplay._complete_parallel_capture(copy(lease))
        gameplay._cancel_parallel_capture(lease)
        with self.assertRaisesRegex(ValueError, 'fresh live'):
            gameplay._complete_parallel_capture(lease)

    def test_wrong_token_slot_or_plan_consumes_lease_without_capture(self):
        original_root, original_binding = self.root, self.binding
        for index, changes in enumerate(({'token': 'foreign'}, {'schedule_index': 1},
                                         {'plan_sha256': 'b'*64}, {'status': 'failed'})):
            with self.subTest(changes=changes):
                self.root = original_root/str(index)
                self.root.mkdir()
                path, plan = fixtures.write_plan(self.root, rows=2)
                self.binding = fixtures.binding_for(self.root, self.model, path, plan, 0)
                self.addCleanup(gameplay._BINDINGS.pop, id(self.binding), None)
                lease = self.lease()
                self.child(lease, message_changes=changes)
                before = set(gameplay._CAPTURES)
                with self.assertRaisesRegex(ValueError, 'live lease'):
                    gameplay._complete_parallel_capture(lease)
                self.assertEqual(set(gameplay._CAPTURES), before)
                with self.assertRaisesRegex(ValueError, 'fresh live'):
                    gameplay._complete_parallel_capture(lease)
        self.root, self.binding = original_root, original_binding

    def test_capture_modified_after_child_completion_is_rejected(self):
        lease = self.lease()
        self.child(lease)
        path = Path(self.binding.folder)/'builtin-training-rollout.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            gameplay._complete_parallel_capture(lease)

    def test_capture_for_another_binding_is_rejected_even_with_matching_child_digest(self):
        lease = self.lease()
        changed = self.binding.to_dict()
        changed['schedule_index'] = 1
        self.child(lease, capture_changes={'binding': changed})
        with self.assertRaisesRegex(ValueError, 'another binding'):
            gameplay._complete_parallel_capture(lease)

    def test_guarded_plan_rejects_unguarded_capture_before_import(self):
        from experiments.learner.interaction_guard import VERSION
        # Freeze a new binding after declaring support, then transfer the same
        # genuine synthetic unguarded decisions through the live child pipe.
        self.plan['policy_support'] = VERSION
        self.path.write_text(json.dumps(self.plan))
        self.binding = fixtures.binding_for(self.root, self.model, self.path, self.plan, 0)
        self.addCleanup(gameplay._BINDINGS.pop, id(self.binding), None)
        lease = self.lease()
        self.child(lease)
        before = set(gameplay._CAPTURES)
        with self.assertRaisesRegex(ValueError, 'policy support differs'):
            gameplay._complete_parallel_capture(lease)
        self.assertEqual(set(gameplay._CAPTURES), before)


if __name__ == '__main__':
    unittest.main()
