"""Filesystem relocation preserves historical identity and rejects redirection."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.linux_artifacts import Evidence


class LinuxArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = '/Users/test/Projects/artifacts/state.json'
        self.actual = self.root/'tree'/self.original.lstrip('/')
        self.actual.parent.mkdir(parents=True)
        self.actual.write_bytes(b'{"historical_path":"/Users/test/original"}\n')
        self.expected = dict(path=self.original, bytes=self.actual.stat().st_size,
                             sha256=hashlib.sha256(self.actual.read_bytes()).hexdigest())
        # The fixture tests filesystem access independently of the pinned real
        # handoff constructor; production callers must verify its manifest.
        self.access = Evidence.__new__(Evidence)
        self.access.extracted = self.root
        self.access.files = {self.original: {
            'original_path': self.original, 'restored_path': 'tree/'+self.original[1:],
            'bytes': self.expected['bytes'], 'sha256': self.expected['sha256']}}

    def test_rebinding_preserves_bytes_and_logical_metadata(self):
        before = self.actual.read_bytes()
        descriptor = {k: dict(self.expected) for k in ('plan', 'report', 'policy', 'resume', 'initial_checkpoint')}
        result = self.access.rebind(descriptor)
        self.assertEqual(self.access.binding(self.original), self.expected)
        self.assertEqual(result['policy']['path'], str(self.actual))
        self.assertEqual(self.actual.read_bytes(), before)
        self.assertEqual(descriptor['policy']['path'], self.original)

    def test_changed_bytes_and_unbound_historical_paths_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unbound historical'):
            self.access.binding('/Users/test/not-in-manifest')
        self.actual.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.access.read(self.original)

    def test_symlink_redirection_fails_even_with_matching_content(self):
        duplicate = self.root/'duplicate.json'
        duplicate.write_bytes(self.actual.read_bytes())
        self.actual.unlink()
        self.actual.symlink_to(duplicate)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            self.access.binding(self.original)


if __name__ == '__main__':
    unittest.main()
