"""The opt-in exact fast path of the manual local runner: reviewed source transition, execution
choices bound in the session and plans, the serial verification fallback, and the runner's
default (the accepted implementation). No native game or PPO update is launched."""
import hashlib
import importlib.util
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from experiments.headless import background_check
from experiments.learner import local_script_training as local
from experiments.learner import ppo

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_629 = '3268f2f8cae6a01277c1629c5d0a9fb8958b5786'
spec = importlib.util.spec_from_file_location('local_controller_fast', ROOT/'tools/run_local_training.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
pins_spec = importlib.util.spec_from_file_location('fast_path_pins', ROOT/'tools/acceleration3/fast_path_pins.py')
pins = importlib.util.module_from_spec(pins_spec)
pins_spec.loader.exec_module(pins)


def accepted_parent(commit=ACCEPTED_629):
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', 'experiments/learner',
                                     'dependencies/learner.lock'], cwd=ROOT, text=True).split()
    names = [n for n in names if (n.endswith('.py') and n.count('/') == 2) or n == 'dependencies/learner.lock']
    digests = {n: hashlib.sha256(subprocess.check_output(['git', 'show', f'{commit}:{n}'], cwd=ROOT)).hexdigest() for n in names}
    return {'interface': local.INTERFACE, 'torch': str(torch.__version__), 'git_commit': commit, 'source_sha256': digests}


class ReviewedTransitionTests(unittest.TestCase):
    def test_pins_equal_the_tree_and_the_accepted_commits(self):
        self.assertEqual({k: (tuple(v[0]), v[1]) for k, v in local.FAST_PATH_SOURCES.items()}, pins.expected_pins())
        self.assertEqual(local.LOCAL_ADAPTER_TRANSITION_AST, pins.adapter_ast())

    def test_accepted_parents_continue_under_the_named_fast_path(self):
        for commit in (ACCEPTED_629, local.ACCEPTED_ADAPTER_COMMIT):
            with self.subTest(commit=commit):
                proof = local.source_compatibility(accepted_parent(commit))
                self.assertEqual(proof['reviewed_exact_fast_path'], sorted(local.FAST_PATH_SOURCES))
                self.assertIs(proof['policy_ppo_unchanged'], False)
                for name in local.FAST_PATH_SOURCES:
                    self.assertEqual(proof['changes'][name]['transition'], 'reviewed_exact_fast_path')

    def test_a_fast_parent_continues_without_transition(self):
        parent = accepted_parent()
        for name in parent['source_sha256']:
            parent['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        parent['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        try:
            proof = local.source_compatibility(parent)
        except ValueError as error:  # an uncommitted tree: the historical bytes cannot be shown
            self.assertIn('not committed', str(error)); return
        self.assertEqual((proof['changes'], proof['reviewed_exact_fast_path'], proof['policy_ppo_unchanged']), ({}, [], True))

    def test_unreviewed_bytes_on_either_side_are_rejected(self):
        name = 'experiments/learner/ppo.py'
        accepted, fast = local.FAST_PATH_SOURCES[name]
        current = (ROOT/name).read_bytes()
        self.assertEqual(local.fast_path_transition(name, accepted[0], current)['current_sha256'], fast)
        for expected, data in [('0'*64, current), (accepted[0], current + b'\n'), (fast, current)]:
            with self.subTest(expected=expected[:8]), self.assertRaisesRegex(ValueError, 'implementation changed'):
                local.fast_path_transition(name, expected, data)
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            local.fast_path_transition('experiments/learner/model.py', accepted[0], current)
        missing = {k: v for k, v in local.FAST_PATH_SOURCES.items() if k != name}
        with patch.object(local, 'FAST_PATH_SOURCES', missing), self.assertRaisesRegex(ValueError, 'implementation changed: ' + name):
            local.source_compatibility(accepted_parent())

    def test_tampered_parent_digest_is_not_committed_history(self):
        parent = accepted_parent()
        parent['source_sha256']['experiments/learner/ppo.py'] = '1'*64
        with self.assertRaisesRegex(ValueError, 'not committed'):
            local.source_compatibility(parent)


class ExecutionChoiceTests(unittest.TestCase):
    def setUp(self):
        saved = background_check.FORK, ppo.PREFIX_HELPERS, ppo.EXPERIMENTS
        def restore():
            background_check.FORK, ppo.PREFIX_HELPERS, ppo.EXPERIMENTS = saved
        self.addCleanup(restore)

    def test_only_known_exact_choices_and_named_experiments(self):
        value = local.execution_choice(overlapped_verification=True, prefix_helpers=3)
        self.assertEqual(value, {'implementation': 'exact-fast-path-v1', 'overlapped_verification': True,
                                 'prefix_helpers': 3, 'experiments': []})
        self.assertEqual(local.execution_choice(overlapped_verification=False, experiments=['shared_graph_encodings'])['experiments'],
                         ['shared_graph_encodings'])
        bad = [dict(value, implementation='accepted'), dict(value, overlapped_verification=1), dict(value, prefix_helpers=9),
               dict(value, prefix_helpers=-1), dict(value, prefix_helpers=True), dict(value, experiments=['faster']),
               dict(value, experiments=['shared_graph_encodings', 'shared_graph_encodings']), dict(value, extra=0),
               {k: v for k, v in value.items() if k != 'experiments'}]
        for item in bad:
            with self.subTest(item=item), self.assertRaisesRegex(ValueError, 'execution choice'):
                local.check_execution(item)

    def test_choices_take_effect_only_from_a_plan(self):
        local.apply_execution({'execution': local.execution_choice(overlapped_verification=False, prefix_helpers=2,
                                                                   experiments=['shared_graph_encodings'])})
        self.assertEqual((background_check.FORK, ppo.PREFIX_HELPERS, ppo.EXPERIMENTS),
                         (False, 2, frozenset({'shared_graph_encodings'})))
        local.apply_execution({'execution': local.execution_choice(overlapped_verification=True)})
        self.assertEqual((background_check.FORK, ppo.PREFIX_HELPERS, ppo.EXPERIMENTS), (True, 0, frozenset()))
        with self.assertRaises(ValueError):
            local.apply_execution({'execution': {'implementation': 'exact-fast-path-v1'}})


def _value(x):
    return x * 2


def _fail(x):
    raise LookupError(x)


class SerialFallbackTests(unittest.TestCase):
    def setUp(self):
        saved = background_check.FORK
        self.addCleanup(setattr, background_check, 'FORK', saved)

    def test_serial_mode_never_forks_and_runs_when_asked(self):
        background_check.FORK = False
        calls = []
        def record(x):
            calls.append(x); return x + 1
        with patch.object(background_check.os, 'fork', side_effect=AssertionError('forked')):
            check = background_check.BackgroundCheck(record, 1).start()
            self.assertEqual(calls, [])  # the caller's own checks run first, as before the overlap
            self.assertEqual(check.result(timeout=1), 2)
            check.stop()
            failing = background_check.BackgroundCheck(_fail, 'x').start()
            with self.assertRaises(LookupError):
                failing.result(timeout=1)
            failing.stop()
        self.assertEqual(calls, [1])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'fork overlap is validated on Linux')
    def test_both_modes_return_the_same_value_and_error(self):
        for fork in (True, False):
            background_check.FORK = fork
            with self.subTest(fork=fork):
                self.assertEqual(background_check.BackgroundCheck(_value, 21).start().result(timeout=10), 42)
                with self.assertRaises(LookupError):
                    background_check.BackgroundCheck(_fail, 'y').start().result(timeout=10)


class RunnerDefaultTests(unittest.TestCase):
    def arguments(self, *extra):
        return ['run_local_training.py', '--config', str(ROOT/'missing.json'), '--profile', 'original', *extra]

    def parse(self, *extra):
        with patch.object(controller.sys, 'argv', self.arguments(*extra)), patch.object(controller, 'start', return_value=0) as start, \
                patch.object(controller.Path, 'exists', return_value=True), patch.object(controller, 'read',
                return_value={'artifact_root': str(ROOT), 'default_profile': 'original'}), patch.dict(controller.os.environ):
            controller.main()
        return start.call_args.args[0]

    def test_default_is_the_accepted_implementation(self):
        args = self.parse('--max-updates', '1')
        self.assertFalse(args.fast_path)
        self.assertEqual((args.prefix_helpers, args.experiment), (0, []))
        self.assertEqual(len(controller.ACCEPTED_IMPLEMENTATION), 40)

    def test_fast_options_require_the_explicit_fast_path(self):
        for extra in [('--prefix-helpers', '2'), ('--experiment', 'shared_graph_encodings'), ('--serial-verification',),
                      ('--overlap-verification',), ('--fast-path', '--prefix-helpers', '9'),
                      ('--fast-path', '--experiment', 'unknown')]:
            with self.subTest(extra=extra), patch.object(controller.sys, 'argv', self.arguments(*extra)), \
                    patch.object(controller, 'start') as start, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    controller.main()
                self.assertEqual(stopped.exception.code, 2)
                start.assert_not_called()

    def test_overlap_default_is_serial_on_macos(self):
        for platform, expected in (('darwin', False), ('linux', True)):
            with self.subTest(platform=platform), patch.object(controller.sys, 'platform', platform):
                self.assertIs(self.parse('--fast-path').overlap_verification, expected)
        self.assertIs(self.parse('--fast-path', '--serial-verification').overlap_verification, False)

    def test_start_releases_accepted_by_default_and_current_only_when_opted_in(self):
        releases = []
        def release(root, commit=None):
            releases.append(commit); return Path('/release'), commit or 'HEAD'
        config = {'artifact_root': '/artifacts', 'run_root': '/artifacts/run', 'default_profile': 'original'}
        for extra, expected, tail in [((), controller.ACCEPTED_IMPLEMENTATION, []),
                                      (('--fast-path', '--serial-verification', '--prefix-helpers', '2'), None,
                                       ['--fast-path', '--prefix-helpers', '2', '--serial-verification'])]:
            with self.subTest(extra=extra):
                args = self.parse(*extra)
                with patch.object(controller, 'read', return_value=config), \
                        patch.dict('sys.modules', {'prepare_local_training': type(sys)('prepare_local_training')}), \
                        patch.object(controller, 'release_checkout', side_effect=release), \
                        patch.object(controller, 'campaign_locks'), patch.object(controller, 'exclusive_lock'), \
                        patch.object(controller.os, 'chdir'), patch.object(controller.os, 'execv') as execv, \
                        patch.dict(controller.os.environ):
                    sys.modules['prepare_local_training'].validate_config = lambda value: value
                    with self.assertRaises(AssertionError):
                        controller.start(args)
                self.assertEqual(releases[-1], expected)
                command = execv.call_args.args[1]
                self.assertEqual(command[-len(tail):] if tail else [x for x in command if x.startswith('--fast')], tail)

    def test_the_fast_release_refuses_a_launch_without_the_explicit_choice(self):
        args = self.parse('--max-updates', '1')
        with patch.object(controller, 'read', return_value={}), \
                patch.dict('sys.modules', {'prepare_local_training': type(sys)('prepare_local_training')}):
            sys.modules['prepare_local_training'].validate_config = lambda value: value
            with self.assertRaisesRegex(ValueError, 'explicit fast path'):
                controller.launch_owned(args)


if __name__ == '__main__':
    unittest.main()
