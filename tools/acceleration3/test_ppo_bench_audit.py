"""Safety checks for the captured-data stress driver; no gameplay or long PPO."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ppo_bench_audit_tool', HERE / 'ppo_bench.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class CapturedStressTests(unittest.TestCase):
    def fixture(self, root):
        batch = root / 'execution'; batch.mkdir()
        args = SimpleNamespace(checkout=root, batch=batch, capture_plan=root / 'plan.json',
            capture_report=batch / 'report.json', require_rolled_back=True)
        parent = {}
        for role in ('plan', 'report', 'policy', 'resume'):
            path = root / ('parent-' + role); path.write_text(role)
            setattr(args, 'parent_' + role if role in ('plan', 'report') else role, path)
            parent[role] = b.file_binding(path)
        game_map = root / 'map'; game_map.write_bytes(b'map')
        plan = dict(mode='training', schedule=[dict(phase='training', episode=i) for i in range(4)],
            parent_descriptor=parent, source={'git_commit':'test', 'source_sha256':{}}, runtime={},
            game_map=str(game_map), map=b.file_binding(game_map))
        args.capture_plan.write_text(json.dumps(plan))
        report = dict(plan_sha256=b.file_binding(args.capture_plan)['sha256'], update_status='rolled_back',
            training_performed=False, training_samples=0, optimizer_steps=0, optimizer_updates=0, attempts=[])
        for i, row in enumerate(plan['schedule']):
            folder = batch / ('episode-' + str(i)); folder.mkdir()
            report['attempts'].append(dict(status='passed', episode_directory=str(folder), weights_sha256='weights',
                **dict.fromkeys(('cleanup_verified','identity_verified','completion_receipt','replay_verified','weights_unchanged'), True)))
            receipt = dict(schema='heroes-builtin-training-receipt-v1', admission_from_file=False, files={},
                binding=dict(schedule_index=i, plan_sha256=report['plan_sha256'], row_json=json.dumps(row),
                    source_json=json.dumps(plan['source']), weights_sha256='weights', checkpoint_file_json=json.dumps(parent['policy'])))
            (folder / 'training-receipt.json').write_text(json.dumps(receipt))
            (folder / 'manifest.json').write_text('{}')
        args.capture_report.write_text(json.dumps(report))
        return args, report

    def test_capture_rejects_any_admission_in_a_required_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            args, report = self.fixture(Path(folder))
            report['training_samples'] = 1
            args.capture_report.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'entirely rolled back'):
                b.audit_capture(args)

    def test_capture_requires_all_verification_files_in_every_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = self.fixture(Path(folder))
            with patch('experiments.learner.builtin_gameplay._capture_file_bindings', return_value={}), \
                 self.assertRaisesRegex(ValueError, 'omits bound evidence'):
                b.audit_capture(args)

    def test_bound_bytes_allow_relocation_but_reject_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = (Path(folder) / n for n in ('first', 'second'))
            first.write_bytes(b'full state'); second.write_bytes(first.read_bytes())
            identity = b.file_binding(first)
            self.assertEqual(b.same_file_bytes(second, identity)['sha256'], identity['sha256'])
            second.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Bound input changed'):
                b.same_file_bytes(second, identity)

    def test_full_state_keeps_all_rng_and_resume_components(self):
        import random
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        generator = torch.Generator().manual_seed(71)
        parent = SimpleNamespace(progress=SimpleNamespace(updates=699, decisions=366475),
            sampler_state={'sequence': 336}, opponent_pool=SimpleNamespace(state_dict=lambda: {'fixed': 'Nullkiller'}))
        before = b.full_state(model, optimizer, generator, parent)
        self.assertEqual(set(before), {'model', 'adam', 'torch_rng', 'python_rng', 'minibatch_rng', 'progress', 'sampler', 'pool'})
        digest = b.tensor_digest(before)
        for key in ('progress', 'sampler', 'pool'):
            self.assertIsNot(before[key], getattr(parent, {'progress':'progress', 'sampler':'sampler_state', 'pool':'opponent_pool'}[key]))
        model.bias.data.add_(1)
        generator.manual_seed(72)
        self.assertEqual(b.tensor_digest(before), digest)
        after = b.full_state(model, optimizer, generator, parent)
        self.assertNotEqual(b.tensor_digest(after), digest)
        self.assertEqual(before['python_rng'], random.getstate())

    def test_guard_owns_a_new_group_and_kills_only_its_unreaped_child(self):
        process = SimpleNamespace(pid=12345)
        process.wait = unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired('diagnostic', .1),
                                                       subprocess.TimeoutExpired('diagnostic', 10), -9])
        with patch.object(b.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(b.os, 'killpg') as kill:
            code, receipt = b._supervise_locked(['--output', '/tmp/unused-test.json'], .1, 99)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual(kill.call_args_list, [unittest.mock.call(12345, b.signal.SIGTERM),
                                              unittest.mock.call(12345, b.signal.SIGKILL)])
        self.assertEqual(code, -9)
        self.assertTrue(receipt['child_reaped'])
        self.assertEqual(receipt['status'], 'timed_out_or_interrupted')
        self.assertAlmostEqual(receipt['deadline_unix'] - receipt['started_unix'], .1, places=5)

    def test_guard_does_not_signal_a_finished_child(self):
        process = SimpleNamespace(pid=12345, wait=unittest.mock.Mock(return_value=0))
        with patch.object(b.subprocess, 'Popen', return_value=process), patch.object(b.os, 'killpg') as kill:
            code, receipt = b._supervise_locked([], 10, 99)
        kill.assert_not_called()
        self.assertEqual((code, receipt['status']), (0, 'finished'))


if __name__ == '__main__':
    unittest.main()
