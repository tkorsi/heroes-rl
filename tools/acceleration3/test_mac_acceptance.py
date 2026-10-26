"""Linux-runnable tests of the Mac acceptance tool's guards (no game, no Torch)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'tools'))
import mac_acceptance as tool  # noqa: E402
import run_local_training as runner  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


class LockTests(unittest.TestCase):
    def test_lock_excludes_the_official_runner_both_ways(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'local-script-training' / 'owner.lock'
            lock.parent.mkdir(); lock.write_text('')
            with mock.patch.object(tool, 'OFFICIAL_LOCK', lock):
                descriptor, state = tool.hold_official_lock()
                self.assertEqual(state, 'held')
                with self.assertRaisesRegex(RuntimeError, 'Another local training script'):
                    with runner.exclusive_lock(lock):
                        pass
                os.close(descriptor)
                with runner.exclusive_lock(lock):
                    with self.assertRaisesRegex(SystemExit, 'official local training lock is held'):
                        tool.hold_official_lock()

    def test_missing_lock_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'owner.lock'
            with mock.patch.object(tool, 'OFFICIAL_LOCK', lock):
                self.assertEqual(tool.hold_official_lock(), (None, 'absent'))
            self.assertFalse(lock.exists())


class ParentCopyTests(unittest.TestCase):
    def fake_run(self, root, names):
        files = {}
        for key, name in names.items():
            files[key] = write(root / 'official' / key / name, (key * 100).encode())
        run = root / 'official' / 'run-x'
        descriptor = {k: tool.binding(v) for k, v in files.items()}
        write(run / 'session.json', dict(run_dir=str(run)))
        write(run / 'campaign-report.json', dict(status='finished', owned_cleanup_verified=True, initial_state_unchanged=True,
                                                  full_final_restore=dict(exact=True), latest_parent=descriptor))
        write(run / 'shutdown.json', dict(owned_cleanup_verified=True, worker_returncode=0, within_hard_deadline=True))
        return run, descriptor

    def test_copies_by_role_and_checks_published_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The official policy and its initial dependency share the basename after-policy.pt.
            run, descriptor = self.fake_run(root, dict(plan='plan.json', report='report.json', policy='after-policy.pt',
                                                       resume='after-resume.pt', initial_checkpoint='after-policy.pt'))
            expected = {k: descriptor[k]['sha256'] for k in ('policy', 'resume')}
            with mock.patch.object(tool, 'EXPECTED_PARENT', expected), \
                    mock.patch.object(tool, 'OFFICIAL', root / 'official'), \
                    mock.patch('prepare_local_training.INPUT_PINS', {}), \
                    mock.patch('prepare_local_training.PREREQUISITE_PINS', {}):
                inputs, official = tool.prepare_inputs(runner, root / 'bench', run)
            copied = json.loads((inputs / 'parent629.json').read_text())['parent']
            self.assertEqual({k: v['sha256'] for k, v in copied.items()}, {k: v['sha256'] for k, v in descriptor.items()})
            self.assertEqual(len({v['path'] for v in copied.values()}), 5)
            self.assertTrue(all(Path(v['path']).is_relative_to(root / 'bench') for v in copied.values()))

    def test_refuses_a_parent_that_is_not_final629(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _ = self.fake_run(root, dict(plan='plan.json', report='report.json', policy='after-policy.pt',
                                              resume='after-resume.pt', initial_checkpoint='dep.pt'))
            with mock.patch.object(tool, 'OFFICIAL', root / 'official'):
                with self.assertRaisesRegex(SystemExit, 'not the published final 629'):
                    tool.prepare_inputs(runner, root / 'bench', run)

    def test_refuses_an_unfinalized_parent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _ = self.fake_run(root, dict(plan='plan.json', report='report.json', policy='after-policy.pt',
                                              resume='after-resume.pt', initial_checkpoint='dep.pt'))
            write(run / 'shutdown.json', dict(owned_cleanup_verified=True, worker_returncode=1, within_hard_deadline=True))
            with self.assertRaisesRegex(ValueError, 'not cleanly finalized'):
                tool.prepare_inputs(runner, root / 'bench', run)


class StopTests(unittest.TestCase):
    def test_stop_is_requested_during_the_last_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'run'
            write(root / 'latest-run.json', dict(run_dir=str(run)))
            log = run / 'research-log.jsonl'
            stop = threading.Event()
            thread = threading.Thread(target=tool.request_stop_during, args=(root, 2, stop)); thread.start()
            lines = [dict(event='batch_intent', mode='baseline'), dict(event='batch_intent', mode='training'),
                     dict(event='batch_result', mode='training')]
            write(log, ''.join(json.dumps(x) + '\n' for x in lines))
            time.sleep(1.5)
            self.assertFalse((run / 'stop-requested').exists())
            with log.open('a') as stream:
                stream.write(json.dumps(dict(event='batch_intent', mode='training')) + '\n')
            thread.join(5); stop.set()
            self.assertTrue((run / 'stop-requested').exists())


class TreeTests(unittest.TestCase):
    def test_tree_finds_descendants(self):
        child = subprocess.Popen([sys.executable, '-c', 'import subprocess,sys,time; '
                                  'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); time.sleep(30)'])
        try:
            time.sleep(1.0)
            found = tool.tree(child.pid)
            self.assertIn(child.pid, found)
            self.assertEqual(len(found), 2)
        finally:
            for pid in tool.tree(child.pid):
                os.kill(pid, 9)
            child.wait()


class CommandTests(unittest.TestCase):
    def test_campaign_flags(self):
        self.assertEqual(tool.CAMPAIGNS['A'], [])
        self.assertIn('--serial-verification', tool.CAMPAIGNS['F'])
        self.assertEqual(tool.EXACT, set('AFOH'))
        self.assertNotIn('X', tool.EXACT)
        parsed = []
        for letter, flags in tool.CAMPAIGNS.items():
            with mock.patch.object(sys, 'argv', ['run_local_training.py', 'start', '--profile', 'original', *flags]), \
                    mock.patch.object(runner, 'start', side_effect=lambda a: parsed.append(a) or 0), \
                    mock.patch.object(Path, 'exists', return_value=True), \
                    mock.patch.object(runner, 'read', return_value=dict(default_profile='original', artifact_root='/x')):
                runner.main()
        self.assertEqual([a.fast_path for a in parsed], [False, True, True, True, True])
        self.assertEqual(parsed[4].experiment, ['shared_graph_encodings'])


if __name__ == '__main__':
    unittest.main()
