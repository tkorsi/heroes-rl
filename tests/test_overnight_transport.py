"""Transport-only recovery never substitutes for checkpoint/game integrity."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
from tools import overnight_transport_watch as w


class TransportControls(unittest.TestCase):
    def test_only_declared_transient_http_failures_qualify(self):
        self.assertTrue(w.transient('RPC failed; HTTP 503 curl22'))
        self.assertTrue(w.transient('The requested URL returned error: 429'))
        for text in ('non-fast-forward','replay mismatch','checksum mismatch','permission denied'):
            self.assertFalse(w.transient(text))

    def test_ambiguous_push_requires_exact_remote_acknowledgement(self):
        head='a'*40
        values=[N(returncode=1,stdout=b'',stderr=b'HTTP 503'),
            N(returncode=0,stdout=(head+'\trefs/heads/x\n').encode(),stderr=b'')]
        with patch.object(w,'checked',return_value=head.encode()),patch.object(w,'git',side_effect=values) as git:
            result=w.push_ack()
            self.assertTrue(result['remote_acknowledged']);self.assertEqual(result['attempts'],1)
            self.assertEqual(git.call_args_list[0].args,('push','origin','HEAD:refs/heads/'+w.BRANCH))

    def test_missing_remote_ack_stops_after_bounded_retries(self):
        def git(*args):
            return N(returncode=1,stdout=b'',stderr=b'HTTP 503')
        with patch.object(w,'checked',return_value=b'a'*40),patch.object(w,'git',side_effect=git),patch.object(w.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'bounded retries'):w.push_ack()

    def test_incomplete_game_or_ppo_is_never_transport_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);report=root/'campaign-report.json'
            report.write_text(json.dumps(dict(status='stopped_failure',error='push failed',batches=[{'checkpoint_verified':False}])))
            (root/'controller-result.json').write_text(json.dumps(dict(owned_cleanup_verified=True,within_budget=True)))
            with patch.object(w,'verify_pair') as restore:
                with self.assertRaisesRegex(ValueError,'No replay/gameplay/PPO/integrity'):w.verify_stopped(report)
                restore.assert_not_called()


if __name__=='__main__':unittest.main()
