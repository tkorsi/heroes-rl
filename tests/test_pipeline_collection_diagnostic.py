"""Diagnostic source/output isolation and distinct seed ownership; no gameplay."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import benchmark_pipeline_collection as diagnostic


class PipelineDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base = self.root/'run-investigations/pipeline-collection'
        patcher = patch.object(diagnostic, 'BASE', self.base)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(diagnostic.copies, 'ARTIFACTS', self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_separate_reservations_never_reuse_consumed_seeds_and_cover_both_seats(self):
        first = diagnostic.reserve_schedule(self.base/'first')
        second = diagnostic.reserve_schedule(self.base/'second')
        keys = ('engine_seed', 'policy_seed')
        left = {row[key] for row in first['schedule'] for key in keys}
        right = {row[key] for row in second['schedule'] for key in keys}
        self.assertEqual(len(left), 16)
        self.assertFalse(left & right)
        self.assertEqual([row['learner_player'] for row in first['schedule']], [0]*4+[1]*4)
        self.assertEqual(first['paired_arms'], [name for name, _ in diagnostic.ARMS])
        self.assertFalse(first['retries_permitted'])
        ledger = json.loads((self.base/'seed-ledger.json').read_text())
        self.assertEqual(ledger['next_study'], 2)

    def test_output_cannot_be_official_state_or_an_existing_attempt(self):
        with self.assertRaises(ValueError):
            diagnostic.diagnostic_output(self.root/'local-script-training/new', fresh=True)
        output = self.base/'existing'; output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            diagnostic.diagnostic_output(output, fresh=True)

    def test_output_cannot_follow_a_symlink(self):
        self.base.mkdir(parents=True)
        (self.base/'redirect').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            diagnostic.diagnostic_output(self.base/'redirect/new', fresh=True)

    def test_changed_copied_inputs_are_rejected_before_a_worker_runs(self):
        path = self.root/'policy.pt'; path.write_bytes(b'original')
        launch = dict(inputs={'policy': diagnostic.binding(path)})
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Copied diagnostic input changed'):
            diagnostic.verify_inputs(launch)

    def test_private_lease_rejects_nonempty_or_replaced_identity(self):
        output = self.base/'lease'; output.mkdir(parents=True)
        path = output/'diagnostic.lock'; path.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'lease file changed'):
            with diagnostic.local_lease(output):
                self.fail('nonempty lease accepted')
        path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'lease identity changed'):
            with diagnostic.local_lease(output):
                path.unlink(); path.touch()

    def test_collector_requires_the_explicit_excluded_scope(self):
        from experiments.learner import parallel_collection as parallel
        args = SimpleNamespace(plan=self.root/'plan.json', sha256='x')
        with patch.object(parallel, '_read_plan', return_value={}), patch.object(parallel, '_collect_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'isolated collector scope'):
                diagnostic.collect_child(args)
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
