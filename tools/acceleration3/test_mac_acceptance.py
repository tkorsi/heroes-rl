"""Linux-runnable tests of the Mac acceptance tool's guards (no game, no Torch)."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
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


class OfficialRootTests(unittest.TestCase):
    def test_paths_inside_the_official_root_are_refused_by_identity_or_folded_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            official = Path(tmp) / 'Heroes-artifacts'
            with mock.patch.object(tool, 'OFFICIAL', official):
                self.assertTrue(tool.inside_official(Path(tmp) / 'heroes-ARTIFACTS' / 'bench'))  # root absent: case-folded
                official.mkdir()
                (Path(tmp) / 'alias').symlink_to(official)
                self.assertTrue(tool.inside_official(Path(tmp) / 'alias' / 'bench' / 'deeper'))
                self.assertTrue(tool.inside_official(official))
                self.assertFalse(tool.inside_official(Path(tmp) / 'Heroes-bench' / 'x'))


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
    def parse(self, *argv, config=None):
        parsed = []
        with mock.patch.object(sys, 'argv', ['run_local_training.py', 'start', '--profile', 'original', *argv]), \
                mock.patch.object(runner, 'start', side_effect=lambda a: parsed.append(a) or 0), \
                mock.patch.object(Path, 'exists', return_value=True), \
                mock.patch.object(runner, 'read', side_effect=lambda path: dict(
                    default_profile='original', artifact_root='/x',
                    run_root='/bench/run' if Path(path) == Path('/bench/machine.json') else '/official/run')), \
                redirect_stderr(io.StringIO()):
            runner.main()
        return parsed[0]

    def test_campaign_flags(self):
        self.assertEqual(tool.CAMPAIGNS['A'], [])
        self.assertIn('--serial-verification', tool.CAMPAIGNS['F'])
        self.assertEqual(tool.EXACT, set('AFOH'))
        self.assertNotIn('X', tool.EXACT)
        parsed = [self.parse('--config', '/bench/machine.json', *flags) for flags in tool.CAMPAIGNS.values()]
        self.assertEqual([a.fast_path for a in parsed], [False, True, True, True, True])
        self.assertEqual(parsed[4].experiment, ['shared_graph_encodings'])

    def test_an_experiment_never_uses_the_official_configuration_or_run_root(self):
        with self.assertRaises(SystemExit):
            self.parse(*tool.CAMPAIGNS['X'])  # default: the official original configuration
        with mock.patch.object(Path, 'resolve', lambda self: self), self.assertRaises(SystemExit):
            self.parse('--config', '/copy/machine.json', *tool.CAMPAIGNS['X'])  # a copy naming the official run root
        self.assertFalse(self.parse(*tool.CAMPAIGNS['F']).experiment)


class MainTests(unittest.TestCase):
    """main() end to end with the game replaced: dry run, verdict and exit status."""
    def run_main(self, tmp, *argv, campaign=None, batches=None):
        saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        self.addCleanup(lambda: (signal.signal(signal.SIGINT, saved[0]), signal.signal(signal.SIGTERM, saved[1])))
        tool.INTERRUPT.update(requested=None, runner=None)
        bench = Path(tmp) / 'bench'
        releases = {}
        def batches_and_release(run_root):
            value = batches(run_root)
            releases[str(run_root)] = value.pop('release')
            return dict(value, run_dir=str(run_root))
        with mock.patch.object(sys, 'platform', 'darwin'), \
                mock.patch.object(tool, 'release_of', side_effect=lambda run_dir: releases[run_dir]), \
                mock.patch.object(sys, 'argv', ['mac_acceptance.py', '--bench-root', str(bench), *argv]), \
                mock.patch.object(tool, 'preflight', return_value=(runner, 'abc')), \
                mock.patch.object(tool, 'hold_official_lock', return_value=(None, 'absent')), \
                mock.patch.object(tool, 'prepare_inputs', return_value=(Path(tmp) / 'inputs', {'policy': {'sha256': 'p'}})), \
                mock.patch.object(tool, 'OFFICIAL_CONFIG', Path(tmp) / 'missing.json'), \
                mock.patch.object(tool, 'git', side_effect=lambda *a: 'abc' if a[0] == 'rev-parse' else ''), \
                mock.patch.object(tool, 'run_campaign', side_effect=campaign), \
                mock.patch.object(tool.campaign_summary, 'batches', side_effect=batches and batches_and_release), \
                redirect_stdout(io.StringIO()):
            code = tool.main()
        return code, json.loads((bench / 'summary.json').read_text())

    def test_dry_run_writes_a_summary_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = self.run_main(tmp, '--dry-run')
            self.assertEqual(code, 0)
            self.assertIs(summary['dry_run'], True)
            self.assertEqual(summary['platform']['sysname'], os.uname().sysname)

    @staticmethod
    def campaign_result(letter, state, returncode=0, status='stopped_requested', states=None, games=None):
        batches = [dict(sequence=629, seat=s, mode='baseline', status='passed', seconds=1.0, games=['g'] * 4,
                        raw=['r'] * 4, choices=[1] * 4) for s in (0, 1)]
        for index, value in enumerate(states or [state]):
            batches.append(dict(sequence=266 + index, seat=index % 2, mode='training', status='passed', seconds=1.0,
                                games=(games or {}).get(index, ['t%d' % index] * 4), raw=['r'] * 4, choices=[1] * 4,
                                state=value))
        run = dict(returncode=returncode, bound='not_reached', leftover_processes=[])
        release = runner.ACCEPTED_IMPLEMENTATION if letter == 'A' else 'abc'
        return run, dict(status=status, batches=batches, run_dir='', release=release)

    def fake(self, results):
        runs = iter(results)
        current = {}
        def campaign(args, bench, inputs, site, letter, index, env, data):
            current['value'] = next(runs)
            return f'{index:02d}-{letter}', Path(bench) / str(index), current['value'][0]
        return campaign, lambda run_root: dict(current['value'][1])

    def test_identical_clean_campaigns_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batches = self.fake([self.campaign_result('A', 's'), self.campaign_result('F', 's')])
            code, summary = self.run_main(tmp, '--order', 'AF', campaign=campaign, batches=batches)
            self.assertEqual((code, summary['passed'], summary['problems']), (0, True, []))

    def verdict(self, order, results, *argv):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batches = self.fake(results)
            return self.run_main(tmp, '--order', order, *argv, campaign=campaign, batches=batches)

    def test_every_exact_campaign_needs_a_comparable_batch(self):
        first, other, second = self.campaign_result('A', 's'), self.campaign_result('F', 's'), self.campaign_result('F', 's')
        other[1]['batches'][2]['games'] = ['different'] * 4
        code, summary = self.verdict('AFF', [first, other, second])
        self.assertEqual(code, 1)
        self.assertEqual(summary['exact_campaign_status'], {'01-F': 'not_comparable', '02-F': 'identical'})
        self.assertEqual(summary['exact_campaigns_identical'], 'not_comparable: 01-F')

    def test_an_accepted_repeat_cannot_stand_in_for_the_fast_path(self):
        results = [self.campaign_result('A', 's'), self.campaign_result('O', 's'), self.campaign_result('O', 's'),
                   self.campaign_result('A', 's')]
        for result in results[1:3]:
            result[1]['batches'][2]['games'] = ['timing'] * 4
        code, summary = self.verdict('AOOA', results)
        self.assertEqual(code, 1)
        self.assertEqual(summary['exact_campaign_status'],
                         {'01-O': 'not_comparable', '02-O': 'not_comparable', '03-A': 'identical'})

    def test_a_later_batch_is_compared_only_while_the_parents_agree(self):
        # 266: different games (so different updates); 267: identical games from different parents.
        a = self.campaign_result('A', None, states=['a266', 'a267'], games={0: ['x'] * 4})
        f = self.campaign_result('F', None, states=['f266', 'f267'], games={0: ['y'] * 4})
        code, summary = self.verdict('AF', [a, f], '--updates', '2')
        self.assertEqual(summary['exact_campaign_status'], {'01-F': 'not_comparable'})
        self.assertNotEqual(summary['exact_campaigns_identical'], False)
        a = self.campaign_result('A', None, states=['s266', 'a267'])
        f = self.campaign_result('F', None, states=['s266', 'f267'])
        code, summary = self.verdict('AF', [a, f], '--updates', '2')
        self.assertEqual((code, summary['exact_campaigns_identical']), (1, False))

    def test_a_missing_state_is_not_called_a_divergence(self):
        a = self.campaign_result('A', None, status='stopped_failure')
        a[1]['batches'][2]['status'] = 'failed'
        code, summary = self.verdict('AF', [a, self.campaign_result('F', 's')])
        self.assertEqual(code, 1)
        self.assertEqual(summary['exact_campaign_status'], {'01-F': 'not_comparable'})
        self.assertTrue(any('0 of 1 training batches admitted' in x for x in summary['problems']))

    def test_the_release_that_ran_is_checked(self):
        f = self.campaign_result('F', 's')
        f[1]['release'] = 'other'
        code, summary = self.verdict('AF', [self.campaign_result('A', 's'), f])
        self.assertEqual(code, 1)
        self.assertTrue(any('ran release other' in x for x in summary['problems']))

    def test_a_failed_or_divergent_campaign_fails(self):
        cases = [([self.campaign_result('A', 's'), self.campaign_result('F', 'other')], 'identical'),
                 ([self.campaign_result('A', 's'), self.campaign_result('F', 's', returncode=1)], 'exit status 1'),
                 ([self.campaign_result('A', 's'), self.campaign_result('F', None)], 'identical'),
                 ([self.campaign_result('A', 's'), self.campaign_result('F', 's', status='stopped_failure')], 'stopped_failure')]
        for results, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                campaign, batches = self.fake(results)
                code, summary = self.run_main(tmp, '--order', 'AF', campaign=campaign, batches=batches)
                self.assertEqual(code, 1)
                self.assertFalse(summary['passed'])
                self.assertTrue(reason == 'identical' and summary['exact_campaigns_identical'] is not True
                                or any(reason in x for x in summary['problems']), summary['problems'])

    def test_an_interrupt_starts_no_further_campaign_and_still_writes_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batches = self.fake([self.campaign_result('A', 's'), self.campaign_result('F', 's')])
            def interrupting(*a):
                value = campaign(*a); tool.interrupted(signal.SIGINT, None); return value
            code, summary = self.run_main(tmp, '--order', 'AF', campaign=interrupting, batches=batches)
            self.assertEqual(code, 1)
            self.assertEqual(list(summary['campaigns']), ['00-A'])
            self.assertIn('interrupted', summary['problems'])


if __name__ == '__main__':
    unittest.main()
