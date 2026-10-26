"""Scoped file digests: one hash per unchanged file inside a check, a fresh hash after any change."""
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from experiments.headless import digest_scope
from experiments.headless.contextual_evidence import file_identity
from experiments.learner.builtin_training_resume import binding


class DigestScopeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="digest-scope-"))
        self.path = self.root / "a.bin"
        self.path.write_bytes(b"first")

    def expected(self, data):
        return hashlib.sha256(data).hexdigest()

    def count_hashes(self):
        return mock.patch.object(digest_scope, "_hash", wraps=digest_scope._hash)

    def test_outside_a_scope_every_call_rehashes(self):
        with self.count_hashes() as hashed:
            for _ in range(3):
                self.assertEqual(file_identity(self.path), {"sha256": self.expected(b"first"), "bytes": 5})
        self.assertEqual(hashed.call_count, 3)

    def test_inside_a_scope_an_unchanged_file_is_hashed_once_and_scopes_never_share(self):
        with self.count_hashes() as hashed:
            with digest_scope.scope():
                with digest_scope.scope():  # nested scopes share the outer memo
                    self.assertEqual(binding(self.path)["sha256"], self.expected(b"first"))
                self.assertEqual(file_identity(self.path)["sha256"], self.expected(b"first"))
            self.assertEqual(hashed.call_count, 1)
            with digest_scope.scope():
                file_identity(self.path)
            self.assertEqual(hashed.call_count, 2)

    def test_any_rewrite_inside_a_scope_is_rehashed(self):
        with digest_scope.scope():
            self.assertEqual(file_identity(self.path)["sha256"], self.expected(b"first"))
            # Same size and a forced-equal mtime: the status-change time still differs.
            stat = self.path.stat()
            time.sleep(0.01)
            self.path.write_bytes(b"other")
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(file_identity(self.path)["sha256"], self.expected(b"other"))
            # Atomic replacement (new inode) and truncation are new keys too.
            replacement = self.root / "b.bin"
            replacement.write_bytes(b"third")
            os.replace(replacement, self.path)
            self.assertEqual(binding(self.path)["sha256"], self.expected(b"third"))
            self.path.write_bytes(b"")
            self.assertEqual(file_identity(self.path), {"sha256": self.expected(b""), "bytes": 0})

    def test_a_change_during_hashing_is_never_memoized(self):
        original = digest_scope._hash
        def racing(path):
            result = original(path)
            Path(path).write_bytes(b"raced!")  # changes size and times after the bytes were read
            return result
        with digest_scope.scope():
            with mock.patch.object(digest_scope, "_hash", side_effect=racing):
                self.assertEqual(file_identity(self.path)["sha256"], self.expected(b"first"))
            self.assertEqual(file_identity(self.path)["sha256"], self.expected(b"raced!"))

    def test_prefetch_fills_only_the_current_scope_and_errors_stay_with_the_check(self):
        big = self.root / "big.bin"
        big.write_bytes(os.urandom(digest_scope._PARALLEL_BYTES + 3))
        missing = self.root / "missing.bin"
        digest_scope.prefetch([big, missing])  # no scope: nothing happens
        with self.count_hashes() as hashed:
            with digest_scope.scope():
                digest_scope.prefetch([big, missing, self.path])
                prefetched = hashed.call_count
                self.assertEqual(file_identity(big)["sha256"], self.expected(big.read_bytes()))
                self.assertEqual(hashed.call_count, prefetched)
                with self.assertRaises(FileNotFoundError):
                    file_identity(missing)
        self.assertEqual(prefetched, 1)  # small files are left to the sequential check

    def test_symlinked_and_relative_paths_share_the_resolved_file(self):
        link = self.root / "link.bin"
        link.symlink_to(self.path)
        with self.count_hashes() as hashed, digest_scope.scope():
            self.assertEqual(file_identity(link)["sha256"], file_identity(self.path)["sha256"])
        self.assertEqual(hashed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
