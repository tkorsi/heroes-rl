"""Bounded private publication controls use disposable local Git remotes only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import linux_publication as publication


class PublicationBoundaries(unittest.TestCase):
    def test_expired_deadline_and_invalid_ref_fail_before_command(self):
        with patch.object(publication.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'deadline exhausted'):
                publication.command(['git', 'status'], time.time() - 1)
            with self.assertRaisesRegex(ValueError, 'private Codex branch'):
                publication.publish_transaction(base_ref='unused', destination_ref='refs/heads/main',
                    prefix='a', files={}, deadline=time.time() + 60, temporary_root='/tmp')
            process.assert_not_called()

    def test_lost_claim_is_refused(self):
        session = {'linux_cloud': {'claim': {'ref': 'refs/heads/codex/claim', 'commit': 'expected'}},
                   'absolute_deadline_unix': time.time() + 60}
        with patch.object(publication, 'ref_sha', return_value='other'):
            with self.assertRaisesRegex(ValueError, 'claim is absent or changed'):
                publication.check_claim(session)

    def test_failed_publication_keeps_pending_without_advancing_or_replacing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); batch = root/'batch'; batch.mkdir()
            plan = batch/'plan.json'; plan.write_text('{}')
            parent = {k: {'path': str(batch/k), 'sha256': '0'*64, 'bytes': 1}
                      for k in ('plan', 'report', 'policy', 'resume', 'initial_checkpoint')}
            session = {'run_root': str(root), 'run_dir': str(root/'run'), 'artifact_root': str(root),
                'linux_cloud': {'claim': {'ref': 'unused'}, 'results_ref': 'unused', 'handoff_id': 'test'},
                'compute_deadline_unix': time.time() + 60}
            with patch.object(publication, 'check_claim'), patch.object(publication, 'publish_transaction',
                    side_effect=ValueError('transport failed')):
                with self.assertRaisesRegex(ValueError, 'transport failed'):
                    publication.publish_checkpoint(session, parent, {'updates': 1335}, root/'backup', plan)
                pending = root/'pending-checkpoint.json'; before = pending.read_bytes()
                self.assertEqual(json.loads(before)['parent'], parent)
                self.assertFalse((root/'latest-checkpoint.json').exists())
                with self.assertRaisesRegex(ValueError, 'already pending'):
                    publication.publish_checkpoint(session, parent, {'updates': 1336}, root/'backup', plan)
                self.assertEqual(pending.read_bytes(), before)

    def test_prefetch_batches_by_size_and_count_deduplicating_exact_oids(self):
        blobs = [(f'{n:040x}', 4) for n in range(1, 6)]
        with patch.object(publication, 'PREFETCH_BYTES', 9), patch.object(publication, 'PREFETCH_OBJECTS', 2), \
                patch.object(publication, 'command') as command:
            self.assertEqual(publication.prefetch_blobs('/unused', blobs + [blobs[0]], 100), 3)
        requests = [call.kwargs['input_data'].decode().splitlines() for call in command.call_args_list]
        self.assertEqual(requests, [[blobs[0][0], blobs[1][0]], [blobs[2][0], blobs[3][0]], [blobs[4][0]]])
        for call in command.call_args_list:
            self.assertTrue(call.kwargs['transfer'])
            self.assertIn('--no-write-fetch-head', call.args[0])
            self.assertIn('fetch.negotiationAlgorithm=noop', call.args[0])

    def test_timeout_removes_owned_transport_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary)/'child.pid'
            code = ('import subprocess,sys,time,signal; from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"]); '
                    'signal.signal(signal.SIGTERM,lambda *args:(p.wait(timeout=1),sys.exit(0))); '
                    'Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(20)')
            with self.assertRaises(subprocess.TimeoutExpired):
                publication.command([sys.executable, '-c', code, str(pid_file)], time.time() + 10.5)
            self.assertTrue(pid_file.exists())
            pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


class LocalRemotePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='heroes-publication-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.seed = self.root/'seed'
        self.remote = self.root/'remote.git'
        self.work = self.root/'temporary'
        self.source = self.root/'source'; self.source.mkdir()
        self.git('init', '-b', 'codex/base', str(self.seed))
        self.git('-C', str(self.seed), 'config', 'user.name', 'Publication Test')
        self.git('-C', str(self.seed), 'config', 'user.email', 'publication@example.invalid')
        (self.seed/'README').write_text('Disposable publication test.\n')
        self.git('-C', str(self.seed), 'add', 'README')
        self.git('-C', str(self.seed), 'commit', '-m', 'Fixture')
        self.git('clone', '--bare', str(self.seed), str(self.remote))
        self.git('-C', str(self.remote), 'config', 'uploadpack.allowFilter', 'true')
        self.git('-C', str(self.remote), 'config', 'uploadpack.allowAnySHA1InWant', 'true')
        self.remote_patch = patch.object(publication, 'REMOTE', self.remote.as_uri())
        self.remote_patch.start(); self.addCleanup(self.remote_patch.stop)
        self.env_patch = patch.dict(os.environ, GIT_AUTHOR_NAME='Publication Test',
            GIT_AUTHOR_EMAIL='publication@example.invalid', GIT_COMMITTER_NAME='Publication Test',
            GIT_COMMITTER_EMAIL='publication@example.invalid')
        self.env_patch.start(); self.addCleanup(self.env_patch.stop)
        self.ref = 'refs/heads/codex/results'
        self.prefix = 'cloud/test/batches/batch974'
        artifact = self.source/'candidate.json'; artifact.write_text('{"updates":1335}\n')
        self.files = {'candidate.json': artifact}

    def git(self, *args):
        result = subprocess.run(['git', *args], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def publish(self, **kwargs):
        return publication.publish_transaction(base_ref='refs/heads/codex/base', destination_ref=self.ref,
            prefix=self.prefix, files=self.files, deadline=time.time() + 60, temporary_root=self.work, **kwargs)

    def recover(self, receipt, **kwargs):
        return publication.readback_transaction(ref=self.ref, commit=receipt['commit'], prefix=self.prefix,
            manifest_sha256=receipt['manifest_sha256'], files=self.files,
            deadline=time.time() + 60, temporary_root=self.work, **kwargs)

    def head(self):
        return self.git('-C', str(self.remote), 'rev-parse', self.ref)

    def receipt(self):
        commit = self.head()
        manifest = self.git('-C', str(self.remote), 'show', f'{commit}:{self.prefix}/manifest.json') + '\n'
        return dict(commit=commit, manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())

    def mutate_remote(self, action):
        clone = self.root/'mutation'
        self.git('clone', '--branch', 'codex/results', self.remote.as_uri(), str(clone))
        action(clone/ self.prefix)
        self.git('-C', str(clone), 'add', '-A')
        self.git('-C', str(clone), 'commit', '-m', 'Adversarial fixture')
        self.git('-C', str(clone), 'push', 'origin', 'HEAD:'+self.ref)

    def test_publish_prefetches_and_streams_every_artifact_without_lazy_reads(self):
        raw = self.root/'raw'; raw.mkdir()
        (raw/'capture').write_bytes(b'visible-state\x00' * 400)
        game_data = self.root/'copyrighted-data'; game_data.mkdir()
        (game_data/'not-for-publication').write_text('not archived')
        (raw/'Data').symlink_to(game_data, target_is_directory=True)
        with patch.object(publication, 'command', wraps=publication.command) as command:
            receipt = self.publish(raw_root=raw)
        self.assertTrue(receipt['fresh_readback_verified'])
        self.assertFalse(receipt['checkpoint_roundtrip_verified'])
        self.assertFalse(receipt['lazy_fetch_during_verification'])
        self.assertEqual(receipt['explicit_blob_fetch_batches'], 2)
        self.assertEqual(set(receipt['raw_inventory']), {'capture'})
        readback_calls = [call for call in command.call_args_list if 'cat-file' in call.args[0]]
        self.assertEqual(len(readback_calls), 3)
        self.assertTrue(all(call.kwargs['local_only'] for call in readback_calls))
        self.assertTrue(all(call.kwargs.get('output') is not None for call in readback_calls[1:]))
        self.assertFalse(any('show' in call.args[0] for call in command.call_args_list))
        destination = self.root/'verified'
        self.recover(receipt, destination=destination)
        self.assertEqual((destination/'candidate.json').read_bytes(), self.files['candidate.json'].read_bytes())
        with tarfile.open(destination/'raw.part00000', 'r:gz') as archive:
            self.assertFalse(any('not-for-publication' in name for name in archive.getnames()))
        self.assertEqual(list(self.work.iterdir()), [])

    def test_successful_push_then_readback_failure_is_recoverable_without_new_commit(self):
        with patch.object(publication, 'readback_transaction', side_effect=TimeoutError('simulated readback timeout')):
            with self.assertRaisesRegex(TimeoutError, 'simulated readback'):
                self.publish()
        receipt = self.receipt(); published = self.head()
        destination = self.root/'recovered'
        result = self.recover(receipt, destination=destination)
        self.assertTrue(result['fresh_readback_verified'])
        self.assertEqual(self.head(), published)
        self.assertEqual((destination/'candidate.json').read_bytes(), self.files['candidate.json'].read_bytes())
        with self.assertRaisesRegex(ValueError, 'label already exists'):
            self.publish()
        self.assertEqual(self.head(), published)

    def test_existing_publication_must_match_preserved_sources_and_exact_identity(self):
        receipt = self.publish()
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.recover({**receipt, 'manifest_sha256': '0'*64})
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.recover({**receipt, 'commit': '0'*40})
        self.files['candidate.json'].write_text('{"updates":1336}')
        with self.assertRaisesRegex(ValueError, 'preserved source bindings'):
            self.recover(receipt)

    def assert_corruption_refused(self, corruption):
        receipt = self.publish()
        def modify(prefix):
            if corruption == 'bytes': (prefix/'candidate.json').write_text('corrupt')
            else: (prefix/'candidate.json').unlink()
        self.mutate_remote(modify)
        receipt['commit'] = self.head()
        target = self.root/'not-restored'
        with self.assertRaisesRegex(ValueError, 'artifact bytes differ|entry set differs'):
            self.recover(receipt, destination=target)
        self.assertFalse(target.exists())

    def test_corrupt_blob_fails_without_destination(self):
        self.assert_corruption_refused('bytes')

    def test_missing_manifest_entry_fails_without_destination(self):
        self.assert_corruption_refused('missing')

    def test_corrupt_raw_archive_is_not_skipped(self):
        raw = self.root/'raw'; raw.mkdir()
        (raw/'capture').write_text('recorded game capture')
        receipt = self.publish(raw_root=raw)
        self.mutate_remote(lambda prefix: (prefix/'raw.part00000').write_bytes(b'corrupt'))
        receipt['commit'] = self.head()
        with self.assertRaisesRegex(ValueError, 'artifact bytes differ: raw.part00000'):
            self.recover(receipt)

    def test_skipped_prefetch_cannot_be_hidden_by_implicit_network_fetch(self):
        receipt = self.publish()
        prefetch = publication.prefetch_blobs
        calls = 0
        def omit_artifacts(repo, blobs, deadline):
            nonlocal calls
            calls += 1
            return prefetch(repo, blobs, deadline) if calls == 1 else 0
        with patch.object(publication, 'prefetch_blobs', side_effect=omit_artifacts):
            with self.assertRaisesRegex(ValueError, 'Private Git operation failed'):
                self.recover(receipt)
        self.assertEqual(calls, 2)

    def test_pair_restore_failure_prevents_destination_and_success_receipt(self):
        for name in ('plan.json', 'report.json', 'after-policy.pt', 'after-resume.pt'):
            path = self.source/name; path.write_text('{}')
            self.files[name] = path
        with patch('experiments.learner.builtin_preservation.verify_pair', side_effect=ValueError('invalid Adam state')) as restore:
            with self.assertRaisesRegex(ValueError, 'invalid Adam state'):
                self.publish()
            restore.assert_called_once()
        receipt = self.receipt(); destination = self.root/'not-restored'
        with patch('experiments.learner.builtin_preservation.verify_pair', side_effect=ValueError('invalid Adam state')):
            with self.assertRaisesRegex(ValueError, 'invalid Adam state'):
                self.recover(receipt, destination=destination)
        self.assertFalse(destination.exists())
        with patch('experiments.learner.builtin_preservation.verify_pair', return_value={}) as restore:
            migration = object()
            result = self.recover(receipt, destination=destination, migration=migration)
            self.assertTrue(result['checkpoint_roundtrip_verified'])
            self.assertIs(restore.call_args.kwargs['migration'], migration)

    def test_readback_never_overwrites_existing_destination(self):
        receipt = self.publish(); destination = self.root/'existing'; destination.mkdir()
        (destination/'keep').write_text('unchanged')
        with patch.object(publication, 'clone') as clone:
            with self.assertRaisesRegex(ValueError, 'destination already exists'):
                self.recover(receipt, destination=destination)
            clone.assert_not_called()
        self.assertEqual((destination/'keep').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
