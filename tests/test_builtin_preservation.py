"""The redundant copy must survive deletion of the canonical checkpoint pair."""
import json
import shutil
import unittest
from unittest.mock import patch

from tests import test_builtin_training_resume as resume_tests
from experiments.learner import builtin_preservation as preservation


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = resume_tests.ParentResumeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture

    def test_backup_restores_after_original_pair_deleted(self):
        f = self.f
        dest = f.root / 'checkpoints/backup'
        result = preservation.preserve(f.plan_path, f.report_path, dest, root=f.root)
        for kind in ('resume', 'policy'):
            (f.run / 'execution' / ('after-' + kind + '.pt')).unlink()
        proof = preservation.verify_pair(dest / 'after-resume.pt', dest / 'after-policy.pt', f.plan, f.report)
        self.assertEqual(proof, result['restore'])
        self.assertTrue((dest / 'preservation-receipt.json').is_file())

    def test_failed_restore_never_publishes_backup(self):
        f = self.f
        dest = f.root / 'checkpoints/backup'
        with patch.object(preservation, 'verify_pair', side_effect=ValueError('corrupt copy')):
            with self.assertRaisesRegex(ValueError, 'corrupt copy'):
                preservation.preserve(f.plan_path, f.report_path, dest, root=f.root)
        self.assertFalse(dest.exists())
        self.assertEqual(list(dest.parent.iterdir()), [])
        self.assertTrue((f.run / 'execution/after-resume.pt').exists())

    def test_outside_root_rejected(self):
        with self.assertRaisesRegex(ValueError, 'durable root'):
            preservation.durable_path('/tmp/not-durable', self.f.root)

    def test_managed_worktree_rejected(self):
        with self.assertRaisesRegex(ValueError, 'managed worktrees'):
            preservation.durable_path('/Users/orsi/.codex/worktrees/a/file', '/Users/orsi/.codex/worktrees/a')
