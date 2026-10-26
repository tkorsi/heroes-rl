"""Offline controls for exact metadata and narrowly scoped transport routing."""
import unittest
from unittest.mock import patch
from tools import overnight_git_api as api
from tools import overnight_transport_watch as watch


class ExactTransportControls(unittest.TestCase):
    def test_commit_metadata_retains_message_epoch_and_timezone(self):
        raw=(b'tree '+b'a'*40+b'\nparent '+b'b'*40+
             b'\nauthor Example <example@example.invalid> 1792980000 +0200'+
             b'\ncommitter Example <example@example.invalid> 1792980001 -0430\n\nExact message\n\n')
        body=api.commit_body(raw)
        self.assertEqual(body['message'],'Exact message\n\n')
        self.assertTrue(body['author']['date'].endswith('+02:00'))
        self.assertTrue(body['committer']['date'].endswith('-04:30'))
        self.assertEqual(body['parents'],['b'*40])

    def test_extended_signed_metadata_is_never_silently_reauthored(self):
        with self.assertRaisesRegex(ValueError,'Signed/extended'):
            api.commit_body(b'tree a\ngpgsig fake\n\nmessage')

    def test_shim_never_intercepts_another_repository_or_branch(self):
        with patch.object(api.os,'execv') as native,patch.object(api,'push') as push:
            api.shim(['-C','/tmp/unrelated','push','origin','HEAD:refs/heads/'+api.BRANCH])
            native.assert_called_once();push.assert_not_called()
        with patch.object(api.os,'execv') as native,patch.object(api,'push') as push:
            api.shim(['-C','/workspace/heroes-overnight','push','origin','HEAD:refs/heads/main'])
            native.assert_called_once();push.assert_not_called()

    def test_authentication_fault_requires_actual_managed_transport_error(self):
        self.assertTrue(watch.publication_fault("fatal: could not read Username for 'https://github.com': No such device or address"))
        for error in ('permission denied','replay mismatch','checkpoint missing','optimizer failed'):
            self.assertFalse(watch.publication_fault(error))


if __name__=='__main__':unittest.main()
