"""The opt-in exact fast path of the manual local runner: reviewed source transition, execution
choices bound in the session and plans, the serial verification fallback, and the runner's
default (the accepted implementation). No native game or PPO update is launched."""
from copy import deepcopy
import hashlib
import importlib.util
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.headless import background_check
from experiments.learner import local_script_training as local
from experiments.learner import ppo
from experiments.learner import interaction_guard

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
                                     'native/learner', 'dependencies/learner.lock'], cwd=ROOT, text=True).split()
    names = [n for n in names if (n.endswith('.py') and n.count('/') == 2)
             or n.startswith('native/learner/') or n == 'dependencies/learner.lock']
    digests = {n: hashlib.sha256(subprocess.check_output(['git', 'show', f'{commit}:{n}'], cwd=ROOT)).hexdigest() for n in names}
    return {'interface': local.INTERFACE, 'torch': str(torch.__version__), 'git_commit': commit, 'source_sha256': digests}


class ReviewedTransitionTests(unittest.TestCase):
    def test_native_headers_are_in_reviewed_pin_inventory(self):
        headers = ('native/learner/replay_tape.h', 'native/learner/text_replay.h')
        committed = pins.at
        def changed_history(commit, name):
            data = committed(commit, name)
            return data+b'\n' if name in headers else data
        with patch.object(pins, 'ACCEPTED', ('10d3d4f5038025a276b33a0a3d81096e5778130b',)), \
                patch.object(pins, 'at', side_effect=changed_history):
            generated = pins.expected_pins()
        for name in headers:
            with self.subTest(name=name):
                accepted, current_digest = generated[name]
                current = (ROOT/name).read_bytes()
                self.assertEqual(current_digest, hashlib.sha256(current).hexdigest())
                with patch.object(local, 'FAST_PATH_SOURCES', generated):
                    self.assertEqual(local.fast_path_transition(name, accepted[0], current)['transition'],
                                     'reviewed_exact_fast_path')
                    with self.assertRaisesRegex(ValueError, 'implementation changed'):
                        local.fast_path_transition(name, accepted[0], current+b'\n')

    def test_pins_equal_the_tree_and_the_accepted_commits(self):
        self.assertEqual({k: (tuple(v[0]), v[1]) for k, v in local.FAST_PATH_SOURCES.items()}, pins.expected_pins())
        self.assertEqual(local.LOCAL_ADAPTER_TRANSITION_AST, pins.adapter_ast())

    def test_accepted_parents_name_the_separate_policy_support_transition(self):
        for commit in pins.ACCEPTED:
            with self.subTest(commit=commit):
                parent = accepted_parent(commit)
                proof = local.source_compatibility(parent)
                changed = {name for name, digest in parent['source_sha256'].items()
                           if name in local.FAST_PATH_SOURCES and digest != local.FAST_PATH_SOURCES[name][1]}
                self.assertEqual(proof['reviewed_exact_fast_path'], sorted(changed - local.POLICY_SUPPORT_SOURCES))
                self.assertEqual(proof['reviewed_policy_support_transition'],
                    sorted((changed & local.POLICY_SUPPORT_SOURCES) | {'experiments/learner/local_script_training.py'}))
                self.assertIs(proof['policy_ppo_unchanged'], False)
                for name in changed:
                    self.assertEqual(proof['changes'][name]['transition'],
                        'reviewed_policy_support_transition' if name in local.POLICY_SUPPORT_SOURCES else 'reviewed_exact_fast_path')

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
        # Later official parents may already contain this exact fast-path file.
        # Its committed hash is then a reviewed origin, not an unknown digest.
        rejected = [('0'*64, current), (accepted[0], current + b'\n')]
        if fast not in accepted:
            rejected.append((fast, current))
        else:
            self.assertEqual(local.fast_path_transition(name, fast, current)['current_sha256'], fast)
        for expected, data in rejected:
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


class ReviewedTreeTests(unittest.TestCase):
    """Collector, game-interface and runner code may differ from a8ac411 only by the reviewed pins."""
    def test_tree_pins_equal_the_tree(self):
        self.assertEqual(local.FAST_PATH_TREE, pins.expected_tree())
        self.assertEqual(local.reviewed_tree(), sorted(local.FAST_PATH_TREE))
        self.assertIn('experiments/headless/builtin_replay.py', local.FAST_PATH_TREE)
        self.assertIn('reviewed_fast_path_tree', local.source_compatibility(accepted_parent()))

    def test_an_unpinned_or_altered_collector_file_is_refused(self):
        name = 'experiments/headless/builtin_replay.py'
        unpinned = {k: v for k, v in local.FAST_PATH_TREE.items() if k != name}
        altered = dict(local.FAST_PATH_TREE, **{name: (local.FAST_PATH_TREE[name][0], '0' * 64)})
        for tree in (unpinned, altered):
            with self.subTest(pinned=name in tree), patch.object(local, 'FAST_PATH_TREE', tree), \
                    self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: ' + name):
                local.source_compatibility(accepted_parent())

    def test_a_changed_accepted_file_or_new_module_is_refused(self):
        original = Path.read_bytes
        def tampered(path):
            data = original(path)
            return data + b'\n' if path.as_posix().endswith('experiments/headless/contextual_replay.py') else data
        self.assertTrue((ROOT/'experiments/headless/contextual_replay.py').is_file())
        with patch.object(Path, 'read_bytes', tampered), \
                self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: experiments/headless/contextual_replay.py'):
            local.reviewed_tree()
        for folder in ('experiments/headless', 'validation/probes'):
            probe = ROOT/folder/'unreviewed_probe_for_test.py'
            probe.write_text('X = 1\n')
            try:
                with self.subTest(folder=folder), self.assertRaisesRegex(
                        ValueError, f'Unreviewed fast-path source: {folder}/unreviewed_probe_for_test.py'):
                    local.reviewed_tree()
            finally:
                probe.unlink()

    def test_the_admission_verifier_is_reviewed(self):
        original = Path.read_bytes
        def relaxed(path):
            data = original(path)
            return data + b'\n' if path.as_posix().endswith('validation/probes/validate_hero_policy.py') else data
        with patch.object(Path, 'read_bytes', relaxed), \
                self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: validation/probes/validate_hero_policy.py'):
            local.reviewed_tree()

    def test_the_adapter_is_bound_by_its_ast_pin(self):
        with patch.object(local, 'LOCAL_ADAPTER_TRANSITION_AST', '0' * 64), self.assertRaisesRegex(ValueError, 'Unreviewed local adapter'):
            local.reviewed_tree()


class GeneratedEnvironmentDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.git('init', '-q')
        self.write('.gitignore', '.venv/\nignored-authored/\n')
        self.write('experiments/kept.py', 'VALUE = 1\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Baseline')
        self.baseline = self.git('rev-parse', 'HEAD').strip()
        self.environment = self.root / 'experiments/explorer/.venv'

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.root), *arguments], text=True)

    def write(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def generated_environment(self):
        self.write('experiments/explorer/.venv/pyvenv.cfg', 'version = 3.12\n')
        self.write('experiments/explorer/.venv/bin/activate_this.py', 'GENERATED = True\n')

    def reviewed(self):
        with patch.object(local, 'ACCEPTED_ADAPTER_COMMIT', self.baseline), \
                patch.object(local, 'REVIEWED_TREE', ('experiments',)), \
                patch.object(local, 'FAST_PATH_TREE', {}), patch.object(local, 'FAST_PATH_SOURCES', {}):
            return local.reviewed_tree(self.root)

    def test_inactive_ignored_generated_environment_does_not_block_authored_source_gate(self):
        self.generated_environment()
        self.assertEqual(local.discovered_reviewed_sources(self.root), {'experiments/kept.py'})
        self.assertEqual(self.reviewed(), [])
        for name in ('experiments/extra.py', 'experiments/ignored-authored/extra.py'):
            path = self.write(name, 'UNREVIEWED = True\n')
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: ' + name):
                    self.reviewed()
            finally:
                path.unlink()

    def test_missing_symlinked_or_unignored_marker_is_not_an_exemption(self):
        self.generated_environment()
        marker = self.environment / 'pyvenv.cfg'
        marker.unlink()
        probe = 'experiments/explorer/.venv/bin/activate_this.py'
        self.assertIn(probe, local.discovered_reviewed_sources(self.root))
        external = self.write('not-a-venv.cfg', 'version = 3.12\n')
        marker.symlink_to(external)
        self.assertIn(probe, local.discovered_reviewed_sources(self.root))
        marker.unlink()
        marker.write_text('version = 3.12\n')
        (self.root / '.gitignore').write_text('ignored-authored/\n')
        self.assertIn(probe, local.discovered_reviewed_sources(self.root))

    def test_tracked_environment_files_remain_in_the_gate(self):
        self.generated_environment()
        probe = 'experiments/explorer/.venv/bin/activate_this.py'
        self.git('add', '-f', probe)
        with self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: ' + probe):
            self.reviewed()
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Tracked environment')
        self.baseline = self.git('rev-parse', 'HEAD').strip()
        self.assertEqual(self.reviewed(), [])
        self.write(probe, 'ALTERED = True\n')
        with self.assertRaisesRegex(ValueError, 'Unreviewed fast-path source: ' + probe):
            self.reviewed()

    def test_active_environment_or_import_path_cannot_be_excluded(self):
        self.generated_environment()
        for target, value in [('prefix', str(self.environment)),
                              ('path', [*sys.path, str(self.environment / 'lib')])]:
            with self.subTest(target=target), patch.object(sys, target, value), \
                    self.assertRaisesRegex(ValueError, 'Generated virtualenv contributes executable source'):
                local.discovered_reviewed_sources(self.root)
        module = SimpleNamespace(__file__=str(self.environment / 'loaded.py'))
        with patch.dict(sys.modules, generated_env_probe=module), \
                self.assertRaisesRegex(ValueError, 'Generated virtualenv contributes executable source'):
            local.discovered_reviewed_sources(self.root)


class ExecutionChoiceTests(unittest.TestCase):
    def setUp(self):
        saved = background_check.FORK, ppo.PREFIX_HELPERS, ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS, ppo.EXPERIMENTS
        def restore():
            background_check.FORK, ppo.PREFIX_HELPERS, ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS, ppo.EXPERIMENTS = saved
        self.addCleanup(restore)
        self.addCleanup(setattr, interaction_guard, 'ACTIVE_VERSION', interaction_guard.ACTIVE_VERSION)

    def test_guard_activation_is_plan_bound_and_old_plans_clear_it(self):
        execution = local.execution_choice(overlapped_verification=False)
        local.apply_execution({'execution': execution, 'policy_support': interaction_guard.VERSION})
        self.assertEqual(interaction_guard.ACTIVE_VERSION, interaction_guard.VERSION)
        local.apply_execution({'execution': execution})
        self.assertIsNone(interaction_guard.ACTIVE_VERSION)
        with self.assertRaisesRegex(ValueError, 'policy support'):
            local.apply_execution({'execution': execution, 'policy_support': 'unreviewed'})

    def test_only_known_exact_choices_and_named_experiments(self):
        value = local.execution_choice(overlapped_verification=True, prefix_helpers=3)
        self.assertEqual(value, {'implementation': 'exact-fast-path-v1', 'overlapped_verification': True,
                                 'prefix_helpers': 3, 'text_leaf_replay': True, 'replay_threads': 3, 'experiments': [],
                                 'collection_workers': 1})
        self.assertIs(local.execution_choice(overlapped_verification=True, text_leaf_replay=False)['text_leaf_replay'], False)
        self.assertEqual(local.execution_choice(overlapped_verification=False, prefix_helpers=2, text_leaf_replay=False)
                         ['replay_threads'], 0)
        self.assertEqual(local.execution_choice(overlapped_verification=False, replay_threads=2)
                         ['replay_threads'], 2)  # threads without forked helpers
        self.assertEqual(local.execution_choice(overlapped_verification=False, experiments=['shared_graph_encodings'])['experiments'],
                         ['shared_graph_encodings'])
        bad = [dict(value, implementation='accepted'), dict(value, overlapped_verification=1), dict(value, prefix_helpers=9),
               dict(value, prefix_helpers=-1), dict(value, prefix_helpers=True), dict(value, experiments=['faster']),
               dict(value, text_leaf_replay=1), {k: v for k, v in value.items() if k != 'text_leaf_replay'},
               dict(value, collection_workers=True), dict(value, collection_workers=8),
               dict(value, collection_workers=0), dict(value, replay_threads=9), dict(value, replay_threads=-1), dict(value, replay_threads=True),
               {k: v for k, v in value.items() if k != 'replay_threads'}, dict(value, text_leaf_replay=False),
               dict(value, experiments=['shared_graph_encodings', 'shared_graph_encodings']), dict(value, extra=0),
               {k: v for k, v in value.items() if k != 'experiments'}]
        for item in bad:
            with self.subTest(item=item), self.assertRaisesRegex(ValueError, 'execution choice'):
                local.check_execution(item)

    def test_historical_execution_keeps_serial_collection_without_rewriting_it(self):
        value = local.execution_choice(overlapped_verification=False, collection_workers=4)
        self.assertEqual(local.check_execution(value)['collection_workers'], 4)
        historical = {k: v for k, v in value.items() if k != 'collection_workers'}
        self.assertIs(local.check_execution(historical), historical)
        self.assertNotIn('collection_workers', historical)

    def test_choices_take_effect_only_from_a_plan(self):
        local.apply_execution({'execution': local.execution_choice(overlapped_verification=False, prefix_helpers=2,
                                                                   text_leaf_replay=False,
                                                                   experiments=['shared_graph_encodings'])})
        self.assertEqual((background_check.FORK, ppo.PREFIX_HELPERS, ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS,
                          ppo.EXPERIMENTS), (False, 2, False, 0, frozenset({'shared_graph_encodings'})))
        local.apply_execution({'execution': local.execution_choice(overlapped_verification=True)})
        self.assertEqual((background_check.FORK, ppo.PREFIX_HELPERS, ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS,
                          ppo.EXPERIMENTS), (True, 0, True, 0, frozenset()))
        local.apply_execution({'execution': local.execution_choice(overlapped_verification=False, replay_threads=3)})
        self.assertEqual((ppo.PREFIX_HELPERS, ppo.TEXT_REPLAY_WORKERS), (0, 3))
        with self.assertRaises(ValueError):
            local.apply_execution({'execution': {'implementation': 'exact-fast-path-v1'}})


class LineageTests(unittest.TestCase):
    exact = local.execution_choice(overlapped_verification=False)
    shared = local.execution_choice(overlapped_verification=False, experiments=['shared_graph_encodings'])

    def test_guarded_parent_cannot_silently_drop_guard(self):
        version = interaction_guard.VERSION
        parent = {'policy_support': version, 'checkpoint_metadata': {'policy_support': version}}
        local.check_lineage({}, self.exact, version)
        local.check_lineage(parent, self.exact, version)
        with self.assertRaisesRegex(ValueError, 'cannot silently'):
            local.check_lineage(parent, self.exact)
        with self.assertRaisesRegex(ValueError, 'plan/metadata'):
            local.check_lineage({'checkpoint_metadata': {'policy_support': version}}, self.exact, version)
        with self.assertRaisesRegex(ValueError, 'Unknown policy support'):
            local.check_lineage({}, self.exact, 'other')

    def test_guard_binding_matches_session_plan_and_checkpoint(self):
        version = interaction_guard.VERSION
        session = {'policy_support': version}
        plan = {'policy_support': version, 'checkpoint_metadata': {'policy_support': version}}
        self.assertEqual(local.check_policy_support_binding(plan, session), version)
        self.assertIsNone(local.check_policy_support_binding({}, {}))
        for changed, other in (({**plan, 'policy_support': None}, session),
                               ({**plan, 'checkpoint_metadata': {}}, session), (plan, {})):
            with self.assertRaisesRegex(ValueError, 'policy support|Policy support'):
                local.check_policy_support_binding(changed, other)

    def test_an_exact_session_refuses_an_experimental_parent(self):
        experimental = {'checkpoint_metadata': {'numerical_experiments': ['shared_graph_encodings']}}
        for parent in ({}, {'checkpoint_metadata': {}}, {'checkpoint_metadata': {'numerical_experiments': []}}):
            local.check_lineage(parent, self.exact)
            local.check_lineage(parent, self.shared)  # an experiment may start from an exact parent
        local.check_lineage(experimental, self.shared)
        with self.assertRaisesRegex(ValueError, 'numerical experiment this session does not run: shared_graph_encodings'):
            local.check_lineage(experimental, self.exact)

    def test_every_batch_and_the_launch_check_the_parent_lineage(self):
        import ast
        def calls(path, function):
            tree = ast.parse(Path(path).read_text())
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
            return {getattr(c.func, 'attr', getattr(c.func, 'id', None)) for c in ast.walk(node) if isinstance(c, ast.Call)}
        for path, function in ((local.__file__, 'prepare_batch'), (local.__file__, '_check_identity'),
                               (ROOT/'tools/run_local_training.py', 'launch_owned')):
            with self.subTest(function=function):
                self.assertIn('check_lineage', calls(path, function))


def _value(x):
    return x * 2


def _fail(x):
    raise LookupError(x)


class RevealedMovementLineageTests(unittest.TestCase):
    def setUp(self):
        self.execution = local.execution_choice(overlapped_verification=False)

    def parent(self, version):
        return {'policy_support': version,
                'checkpoint_metadata': {'policy_support': version, 'numerical_experiments': []}}

    def test_reviewed_old_to_revealed_transition_leaves_parent_metadata_intact(self):
        parent = self.parent(interaction_guard.VERSION)
        before = deepcopy(parent)
        local.check_lineage(parent, self.execution, interaction_guard.REVEALED_VERSION)
        self.assertEqual(parent, before)
        self.assertEqual(local.check_policy_support(interaction_guard.REVEALED_VERSION), interaction_guard.REVEALED_VERSION)

    def test_revealed_parent_cannot_downgrade_or_drop_support(self):
        parent = self.parent(interaction_guard.REVEALED_VERSION)
        local.check_lineage(parent, self.execution, interaction_guard.REVEALED_VERSION)
        for target in (None, interaction_guard.VERSION):
            with self.subTest(target=target), self.assertRaises(ValueError):
                local.check_lineage(parent, self.execution, target)

    def test_historical_unset_and_v1_lineages_still_work(self):
        local.check_lineage({}, self.execution, None)
        local.check_lineage({}, self.execution, interaction_guard.VERSION)
        local.check_lineage(self.parent(interaction_guard.VERSION), self.execution, interaction_guard.VERSION)
        with self.assertRaises(ValueError):
            local.check_lineage(self.parent(interaction_guard.VERSION), self.execution, None)

    def test_unknown_inherited_or_selected_versions_are_rejected(self):
        for unknown in ('visible-interaction-budget-v0', 'visible-interaction-budget-v2',
                        'unreviewed', '', False):
            with self.subTest(version=unknown):
                with self.assertRaisesRegex(ValueError, 'Unknown policy support'):
                    local.check_policy_support(unknown)
                with self.assertRaises(ValueError):
                    local.check_lineage(self.parent(unknown), self.execution, interaction_guard.REVEALED_VERSION)
                with self.assertRaises(ValueError):
                    local.check_lineage(self.parent(interaction_guard.VERSION), self.execution, unknown)

    def test_transition_cannot_hide_disagreeing_parent_bindings(self):
        for declared, inherited in ((interaction_guard.VERSION, interaction_guard.REVEALED_VERSION),
                                    (interaction_guard.REVEALED_VERSION, interaction_guard.VERSION),
                                    (None, interaction_guard.VERSION)):
            with self.subTest(declared=declared, inherited=inherited):
                parent = self.parent(inherited)
                parent['policy_support'] = declared
                with self.assertRaisesRegex(ValueError, 'plan/metadata'):
                    local.check_lineage(parent, self.execution, interaction_guard.REVEALED_VERSION)

    def test_new_session_plan_and_checkpoint_must_bind_same_version(self):
        for version in interaction_guard.SUPPORTED_VERSIONS:
            plan = self.parent(version)
            session = {'policy_support': version}
            self.assertEqual(local.check_policy_support_binding(plan, session), version)
            other = next(item for item in interaction_guard.SUPPORTED_VERSIONS if item != version)
            with self.assertRaises(ValueError):
                local.check_policy_support_binding(plan, {'policy_support': other})
            altered = deepcopy(plan)
            altered['checkpoint_metadata']['policy_support'] = other
            with self.assertRaises(ValueError):
                local.check_policy_support_binding(altered, session)

    def test_revealed_upgrade_does_not_erase_numerical_experiment_lineage(self):
        parent = self.parent(interaction_guard.VERSION)
        parent['checkpoint_metadata']['numerical_experiments'] = ['shared_graph_encodings']
        with self.assertRaisesRegex(ValueError, 'numerical experiment'):
            local.check_lineage(parent, self.execution, interaction_guard.REVEALED_VERSION)


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
        self.assertIsNone(args.interaction_guard)

    def test_guard_requires_explicit_release_and_survives_cli(self):
        args = self.parse('--fast-path', '--interaction-guard', interaction_guard.VERSION)
        self.assertEqual(args.interaction_guard, interaction_guard.VERSION)
        with patch.object(controller.sys, 'argv', self.arguments('--interaction-guard', interaction_guard.VERSION)), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            controller.main()
        self.assertEqual(stopped.exception.code, 2)

    def test_fast_options_require_the_explicit_fast_path(self):
        for extra in [('--prefix-helpers', '2'), ('--experiment', 'shared_graph_encodings'), ('--serial-verification',),
                      ('--overlap-verification',), ('--collection-workers', '4'),
                      ('--fast-path', '--collection-workers', '8'), ('--no-text-replay',), ('--replay-threads', '2'),
                      ('--fast-path', '--replay-threads', '9'), ('--fast-path', '--no-text-replay', '--replay-threads', '1'),
                      ('--fast-path', '--prefix-helpers', '9'),
                      ('--fast-path', '--experiment', 'unknown')]:
            with self.subTest(extra=extra), patch.object(controller.sys, 'argv', self.arguments(*extra)), \
                    patch.object(controller, 'start') as start, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    controller.main()
                self.assertEqual(stopped.exception.code, 2)
                start.assert_not_called()

    def test_parallel_collection_is_explicit_and_preserves_four_game_contract(self):
        self.assertEqual(self.parse('--fast-path').collection_workers, 1)
        self.assertEqual(self.parse('--fast-path', '--collection-workers', '4').collection_workers, 4)

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
                                      (('--fast-path', '--serial-verification', '--collection-workers', '4'), None,
                                       ['--fast-path', '--prefix-helpers', '0', '--collection-workers', '4',
                                        '--serial-verification', '--replay-threads', '0']),
                                      (('--fast-path', '--serial-verification', '--prefix-helpers', '2'), None,
                                       ['--fast-path', '--prefix-helpers', '2', '--collection-workers', '1', '--serial-verification',
                                        '--replay-threads', '2']),
                                      (('--fast-path', '--serial-verification', '--replay-threads', '3'), None,
                                       ['--fast-path', '--prefix-helpers', '0', '--collection-workers', '1', '--serial-verification',
                                        '--replay-threads', '3']),
                                      (('--fast-path', '--serial-verification', '--no-text-replay'), None,
                                       ['--fast-path', '--prefix-helpers', '0', '--collection-workers', '1', '--serial-verification',
                                        '--no-text-replay'])]:
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
