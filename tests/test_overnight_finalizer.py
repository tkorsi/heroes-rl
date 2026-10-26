"""Final heldout pair only starts after clean shutdown and measured reserve."""
import unittest
from tools import overnight_finalizer as f


class FinalReserveControls(unittest.TestCase):
    def test_absolute_compute_cutoff_never_extends(self):
        journal=dict(status='finished_budget_reserve',batches=[])
        self.assertTrue(f.sufficient_final_reserve(journal,f.COMPUTE-121))
        for now in (f.COMPUTE-120,f.COMPUTE,f.HARD):
            self.assertFalse(f.sufficient_final_reserve(journal,now))

    def test_running_or_failed_controller_never_gets_final_gameplay(self):
        for status in ('running','stopped_failure','preparing'):
            self.assertFalse(f.sufficient_final_reserve(dict(status=status,batches=[]),f.COMPUTE-3600))

    def test_measured_both_seat_and_preservation_reserve(self):
        row=dict(mode='periodic',status='passed',started_utc='2026-10-26T05:00:00+00:00',finished_utc='2026-10-26T05:01:30+00:00')
        journal=dict(status='finished_budget_reserve',batches=[row])
        self.assertFalse(f.sufficient_final_reserve(journal,f.COMPUTE-240))
        self.assertTrue(f.sufficient_final_reserve(journal,f.COMPUTE-241))


if __name__=='__main__':unittest.main()
