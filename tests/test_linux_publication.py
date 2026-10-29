"""Publication failure preserves candidates and never advances the official pointer."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import linux_publication as publication


class PrivatePublicationTests(unittest.TestCase):
    def test_expired_deadline_and_non_private_ref_fail_before_any_command(self):
        with patch.object(publication.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'deadline exhausted'):
                publication.command(['git','status'], time.time()-1)
            with self.assertRaisesRegex(ValueError, 'private Codex branch'):
                publication.publish_transaction(base_ref='unused', destination_ref='refs/heads/main',
                    prefix='a', files={}, deadline=time.time()+60, temporary_root='/tmp')
            run.assert_not_called()

    def test_lost_claim_is_refused(self):
        session={'linux_cloud':{'claim':{'ref':'refs/heads/codex/claim','commit':'expected'}},
                 'absolute_deadline_unix':time.time()+60}
        with patch.object(publication,'ref_sha',return_value='other'):
            with self.assertRaisesRegex(ValueError,'claim is absent or changed'):
                publication.check_claim(session)

    def test_failed_publication_keeps_candidate_without_advancing_or_replacing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); batch=root/'batch';batch.mkdir()
            plan=batch/'plan.json';plan.write_text('{}')
            parent={k:{'path':str(batch/k),'sha256':'0'*64,'bytes':1}
                    for k in ('plan','report','policy','resume','initial_checkpoint')}
            session={'run_root':str(root),'run_dir':str(root/'run'),'artifact_root':str(root),
                     'linux_cloud':{'claim':{'ref':'unused'},'results_ref':'unused','handoff_id':'test'},
                     'compute_deadline_unix':time.time()+60}
            with patch.object(publication,'check_claim'), patch.object(publication,'publish_transaction',side_effect=ValueError('transport failed')):
                with self.assertRaisesRegex(ValueError,'transport failed'):
                    publication.publish_checkpoint(session,parent,{'updates':1230},root/'backup',plan)
                pending=root/'pending-checkpoint.json'
                before=pending.read_bytes()
                self.assertEqual(json.loads(before)['parent'],parent)
                self.assertFalse((root/'latest-checkpoint.json').exists())
                with self.assertRaisesRegex(ValueError,'already pending'):
                    publication.publish_checkpoint(session,parent,{'updates':1231},root/'backup',plan)
                self.assertEqual(pending.read_bytes(),before)


if __name__=='__main__':unittest.main()
