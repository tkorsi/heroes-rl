"""Readiness gates cannot publish an active, shortened, or unverified run."""
import unittest
from tools.finalize_curriculum_handoff import validate_final

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.session=dict(started_unix=100,started_monotonic=100,compute_deadline_unix=7000,absolute_deadline_unix=7300)
        self.controller={'owned_cleanup_verified':True}
        self.journal={'status':'finished_budget_reserve','new_updates':10}
    def test_finalized_window_is_eligible(self):
        validate_final(self.session,self.controller,self.journal,7100,7100)
    def test_running_campaign_is_rejected(self):
        self.journal['status']='running'
        with self.assertRaisesRegex(ValueError,'still active'):validate_final(self.session,self.controller,self.journal,7100,7100)
    def test_cannot_shorten_for_cloud(self):
        with self.assertRaisesRegex(ValueError,'Do not shorten'):validate_final(self.session,self.controller,self.journal,6000,6000)
    def test_unverified_shutdown_and_no_training_are_rejected(self):
        with self.assertRaises(ValueError):validate_final(self.session,{'owned_cleanup_verified':False},self.journal,7100,7100)
        self.journal['new_updates']=0
        with self.assertRaises(ValueError):validate_final(self.session,self.controller,self.journal,7100,7100)
    def test_changed_clock_is_rejected(self):
        self.session['absolute_deadline_unix']+=60
        with self.assertRaisesRegex(ValueError,'window changed'):validate_final(self.session,self.controller,self.journal,7100,7100)
