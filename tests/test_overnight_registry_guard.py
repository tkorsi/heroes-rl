"""Cutoff identities cannot signal a reused or unrelated process."""
import signal
import unittest
from unittest.mock import patch
from tools import overnight_registry_guard as g


class RegistryControls(unittest.TestCase):
    def test_reused_pid_and_changed_command_are_not_owned(self):
        row=dict(pid=17,start_ticks=100,command=['python','owned.py'])
        for actual in (None,dict(row,start_ticks=101),dict(row,command=['other'])):
            with patch.object(g,'snapshot',return_value=actual),patch.object(g.os,'kill') as kill:
                self.assertFalse(g.terminate(row,signal.SIGTERM));kill.assert_not_called()

    def test_exact_identity_can_be_stopped_without_environ(self):
        row=dict(pid=17,start_ticks=100,command=['python','owned.py'])
        with patch.object(g,'snapshot',return_value=row),patch.object(g.os,'kill') as kill:
            self.assertTrue(g.terminate(row,signal.SIGKILL));kill.assert_called_once_with(17,signal.SIGKILL)

    def test_only_compute_worker_or_poller_role_inherits_compute_cutoff(self):
        self.assertTrue(g.compute_command(['python','/release/tools/overnight_curriculum_campaign.py','worker']))
        self.assertFalse(g.compute_command(['python','/release/tools/overnight_curriculum_campaign.py','recover']))
        self.assertTrue(g.compute_command(['python','-m','experiments.learner.overnight_training','worker']))
        self.assertFalse(g.compute_command(['python','/release/tools/overnight_finalizer.py']))


if __name__=='__main__':unittest.main()
