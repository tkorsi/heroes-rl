"""Offline finalization-watcher guards; no process or handoff is launched."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import watch_curriculum_finalization as w

class WatchTests(unittest.TestCase):
    def fixture(self,base):
        root=base/'campaign-v3';root.mkdir()
        clock=dict(started_unix=0,started_monotonic=0,compute_deadline_unix=6900,absolute_deadline_unix=7200)
        window=root/'window.json';window.write_text(json.dumps(clock))
        (base/'campaign-status.json').write_text(json.dumps({'actual_session_path':str(root/'session.json')}))
        (root/'session.json').write_text(json.dumps(dict(clock,window=w.binding(window),controller_pid=1)))
        (root/'controller-result.json').write_text(json.dumps({'finalized_utc':'finished','private_final_preservation':{'commit':'x'},'owned_cleanup_verified':True}))
        (root/'campaign-report.json').write_text(json.dumps({'status':'finished_budget_reserve','new_updates':10}))
        return root,window,clock
    def test_waits_for_real_controller_exit_and_original_finalization_period(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,window,clock=self.fixture(base)
            with patch.object(w,'BASE',base),patch.object(w,'WINDOW',window),patch.object(w,'left',return_value=100),patch.object(w.time,'time',return_value=7100),patch.object(w.time,'monotonic',return_value=7100),patch.object(w.subprocess,'run',return_value=SimpleNamespace(stdout='python curriculum_campaign.py launch')) as ps:
                self.assertIsNone(w.candidate(clock))
                ps.return_value=SimpleNamespace(stdout='')
                self.assertEqual(w.candidate(clock),root)
                with patch.object(w,'left',return_value=1000):self.assertIsNone(w.candidate(clock))
    def test_unfinished_preservation_is_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,window,clock=self.fixture(base)
            (root/'controller-result.json').write_text('{}')
            with patch.object(w,'BASE',base),patch.object(w,'WINDOW',window):self.assertIsNone(w.candidate(clock))
    def test_duplicate_launch_fails_before_process_creation(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,window,clock=self.fixture(base)
            with patch.object(w,'WINDOW',window),patch.object(w,'OUT',root),patch.object(w,'left',return_value=1000),patch.object(w,'guarded_popen') as popen:
                with self.assertRaises(FileExistsError):w.launch()
                popen.assert_not_called()
    def test_bound_helper_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'helper.py';p.write_text('original');manifest={'code':{'helper.py':w.binding(p)}};p.write_text('changed')
            with patch.object(w,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'source drift'):w.code_check(manifest)
