"""Offline manual-session caps preserve original campaign accounting and ownership."""
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.learner import builtin_first_run as pilot
from experiments.learner import inference
from experiments.learner import local_script_training as adapter
from tools import recover_interrupted_training as recovery


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('session_cap_runner', ROOT / 'tools/run_local_training.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SessionGameCapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run_root = self.root / 'runs'
        self.config = dict(artifact_root=str(self.root), run_root=str(self.run_root),
                          default_profile='original', parent={}, runtime_bin='synthetic',
                          runtime={}, data={}, maps={})
        self.config_path = self.root / 'machine.json'
        self.config_path.write_text(json.dumps(self.config))
        self.anchor = self.root / 'anchor'
        self.continuation = dict(target_games=5000, admitted_before_session=2400,
                                 remaining_before_session=2600, next_sequence=869)
        self.state = SimpleNamespace(progress=SimpleNamespace(updates=1229), optimizer_steps=1229)
        self.preparation = SimpleNamespace(validate_config=lambda value: value)

    def argv(self, *extra, continuation=True):
        return ['run_local_training.py', '--config', str(self.config_path), '--profile', 'original',
                '--fast-path', *(['--continue-original', str(self.anchor)] if continuation else []), *extra]

    def arguments(self, *extra):
        with patch.object(sys, 'argv', self.argv(*extra)), patch.dict(runner.os.environ), \
                patch.object(runner, 'start', return_value=0) as start:
            self.assertEqual(runner.main(), 0)
        return start.call_args.args[0]

    def test_cap_requires_original_continuation_and_complete_batches(self):
        cases = [(('--session-games', '1000'), False), (('--session-games', '0'), True),
                 (('--session-games', '-4'), True), (('--session-games', '999'), True),
                 (('--session-games', '4.0'), True), (('--session-games', '1000', '--games', '1000'), True),
                 (('--session-games', '1000', '--max-updates', '250'), True)]
        for extra, continuation in cases:
            with self.subTest(extra=extra), patch.object(sys, 'argv', self.argv(*extra, continuation=continuation)), \
                    patch.object(runner, 'start') as start, patch.object(runner, 'read') as read, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()
            read.assert_not_called()
        args = self.arguments('--session-games', '1000')
        self.assertEqual(args.session_games, 1000)
        self.assertIsNone(args.games)
        self.assertEqual(args.eval_every, 100)

    def test_target_selection_never_silently_truncates_or_resets_original(self):
        self.assertEqual(runner.continuation_game_target(2600, None), 2600)
        self.assertEqual(runner.continuation_game_target(2600, 1000), 1000)
        self.assertEqual(runner.continuation_game_target(1000, 1000), 1000)
        for invalid in (True, 1000.0, 0, -4, 999):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'positive multiple of four'):
                runner.continuation_game_target(2600, invalid)
        with self.assertRaisesRegex(ValueError, 'only 996 remain'):
            runner.continuation_game_target(996, 1000)

    def start_patches(self, stack, remaining=2600):
        stack.enter_context(patch.dict(runner.os.environ))
        stack.enter_context(patch.dict(sys.modules, {'prepare_local_training': self.preparation}))
        stack.enter_context(patch.object(runner, 'campaign_locks', return_value=nullcontext()))
        stack.enter_context(patch.object(runner, 'exclusive_lock', return_value=nullcontext()))
        stack.enter_context(patch.object(runner, 'original_continuation',
                                        return_value=({}, remaining, self.continuation)))

    def test_frozen_release_receives_cap_and_keeps_original_target_argument(self):
        args = self.arguments('--session-games', '1000', '--hours', '48')
        release = self.root / 'frozen-release'
        class Reentered(BaseException):
            pass
        with ExitStack() as stack:
            self.start_patches(stack)
            stack.enter_context(patch.object(runner, 'release_checkout', return_value=(release, 'f'*40)))
            stack.enter_context(patch.object(runner.os, 'chdir'))
            reenter = stack.enter_context(patch.object(runner.os, 'execv', side_effect=Reentered))
            with self.assertRaises(Reentered):
                runner.start(args)
        command = reenter.call_args.args[1]
        self.assertEqual(command[1:3], [str(release / 'tools/run_local_training.py'), '_launch'])
        self.assertEqual(command[command.index('--session-games')+1], '1000')
        self.assertEqual(command[command.index('--continue-original')+1], str(self.anchor))
        self.assertNotIn('--games', command)

    def test_excess_cap_refused_before_frozen_release_or_session_creation(self):
        args = self.arguments('--session-games', '1000')
        with ExitStack() as stack:
            self.start_patches(stack, remaining=996)
            release = stack.enter_context(patch.object(runner, 'release_checkout'))
            supervise = stack.enter_context(patch.object(runner, 'supervise'))
            restore = stack.enter_context(patch.object(adapter, 'restore_parent'))
            reserve = stack.enter_context(patch.object(runner, 'reserve_sequence'))
            stack.enter_context(patch.object(runner, 'selected_execution', return_value={}))
            stack.enter_context(patch.object(adapter, 'check_policy_support', return_value=None))
            for launch in (runner.start, runner.launch_owned):
                with self.subTest(launch=launch.__name__), self.assertRaisesRegex(ValueError, 'only 996 remain'):
                    launch(args)
        for operation in (release, supervise, restore, reserve):
            operation.assert_not_called()
        self.assertFalse(self.run_root.exists())

    def test_session_records_cap_separately_from_unchanged_lineage(self):
        args = self.arguments('--session-games', '1000')
        execution = dict(overlapped_verification=False, collection_workers=4,
                         prefix_helpers=0, text_leaf_replay=False, experiments=[])
        with ExitStack() as stack:
            self.start_patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(runner, 'selected_execution', return_value=execution))
            stack.enter_context(patch.object(adapter, 'check_policy_support', return_value=None))
            stack.enter_context(patch.object(adapter, 'restore_parent', return_value=self.state))
            stack.enter_context(patch.object(adapter, 'read_descriptor', return_value=({}, {})))
            lineage = stack.enter_context(patch.object(adapter, 'check_lineage'))
            stack.enter_context(patch.object(adapter, 'operational_configuration', return_value={'preserved': True}))
            stack.enter_context(patch.object(inference, 'source_identity', return_value={'tracked_worktree_dirty': False}))
            stack.enter_context(patch.object(pilot, 'implementation', return_value={}))
            stack.enter_context(patch.object(pilot, 'prerequisites', return_value={}))
            supervise = stack.enter_context(patch.object(runner, 'supervise', return_value=0))
            self.assertEqual(runner.launch_owned(args), 0)
        session = runner.read(supervise.call_args.args[0])
        self.assertEqual(session['target_games'], 1000)
        self.assertEqual(session['original_continuation'], self.continuation)
        self.assertEqual(session['next_sequence'], 869)
        self.assertEqual(session['configuration'], {'preserved': True})
        self.assertIsNone(args.games)
        lineage.assert_called_once()
        self.assertFalse((self.run_root / 'sequence-ledger.json').exists())

    def test_check_reports_both_targets_and_still_restores_exact_state(self):
        with patch.object(sys, 'argv', self.argv('check', '--session-games', '1000')), \
                patch.dict(sys.modules, {'prepare_local_training': self.preparation}), patch.dict(runner.os.environ), \
                patch.object(runner, 'selected_execution', return_value={}), \
                patch.object(recovery, 'existing_campaign_locks', return_value=nullcontext()), \
                patch.object(runner, 'original_continuation', return_value=({}, 2600, self.continuation)), \
                patch.object(adapter, 'restore_parent', return_value=self.state) as restore, \
                patch.object(adapter, 'read_descriptor', return_value=({}, {})), \
                patch.object(adapter, 'check_lineage') as lineage, \
                patch.object(adapter, 'operational_configuration', return_value={
                    'shared_wall_seconds': 10800, 'phase_deadlines_seconds': {}}), \
                patch.object(runner, 'start') as start, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['remaining_training_games'], 2600)
        self.assertEqual(result['original_remaining_training_games'], 2600)
        self.assertEqual(result['session_target_games'], 1000)
        self.assertFalse(result['gameplay_performed'])
        self.assertEqual(result['ppo_updates'], 0)
        restore.assert_called_once()
        lineage.assert_called_once()
        start.assert_not_called()
        self.assertFalse(self.run_root.exists())

    def test_worker_stops_after_exactly_250_admitted_updates_excluding_evaluation(self):
        run = self.root / 'mock-session'
        run.mkdir()
        session = dict(run_dir=str(run), run_root=str(self.run_root), initial_parent={},
                       supervisor_pid=runner.os.getppid(), profile='original', target_games=1000,
                       original_continuation=self.continuation, next_sequence=869, evaluation_interval_updates=100)
        session_path = run / 'session.json'
        session_path.write_text(json.dumps(session))
        values = []
        def execute(session_path, parent, sequence, seat, mode, value):
            row = dict(mode=mode, status='passed', owned_cleanup_verified=True,
                       outcomes=['loss']*4, checkpoint_verified=mode == 'training')
            value['batches'].append(row)
            if mode == 'training':
                value['latest_progress'] = {'updates': value['latest_progress']['updates']+1}
            return parent
        with patch.object(adapter, 'restore_parent', return_value=self.state), \
                patch.object(runner, 'execute_batch', side_effect=execute), \
                patch.object(runner, 'reserve_sequence', side_effect=range(869, 1119)) as reserve, \
                patch.object(runner, 'enough', return_value=True), patch.object(runner, 'left', return_value=3600), \
                patch.object(runner, 'save', side_effect=lambda run, value: values.append(value)), \
                patch.object(runner, 'event'), patch.object(runner, 'write_markdown'), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.worker(session_path), 0)
        self.assertEqual(reserve.call_count, 250)
        summary = runner.summarize(values[-1])
        self.assertEqual(summary['admitted_training_games'], 1000)
        self.assertEqual(summary['new_ppo_updates'], 250)
        self.assertEqual(summary['evaluation_games'], 32)
        self.assertTrue(summary['target_games_met'])
        self.assertEqual(summary['remaining_training_games'], 0)
        self.assertEqual(summary['stop_reason'], 'training_game_target_reached')
        self.assertEqual(summary['original_campaign'], dict(target_games=5000, admitted_games=3400, remaining_games=1600))


if __name__ == '__main__':
    unittest.main()
