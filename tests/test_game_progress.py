"""Progress is a disposable UI channel, never admission or policy state."""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.learner.game_progress import (GameProgress, ProgressDisplay, color_text, format_progress,
    progress_path, set_phase, terminal_color_enabled, terminal_lines)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = progress_path(self.root, 1)
        self.now = 0.0
        self.progress = GameProgress(self.path, episode=1, ordinary=10, reserved=2, clock=lambda: self.now)

    def read(self):
        return json.loads(self.path.read_text())

    def test_throttles_counts_but_preserves_final_count_and_reserved_budget(self):
        self.progress((0, 0))
        first = self.path.read_bytes()
        for count in range(1, 10):
            self.now += .05
            self.progress.observe_day(0, count, max_days=365)
            self.progress((count, 2))
        self.assertEqual(self.path.read_bytes(), first)
        self.now = 1
        self.progress((10, 2))
        self.assertEqual(self.read()['choices'], [10, 2])
        self.assertEqual(self.read()['days'], [9, None])
        self.progress((11, 2), 'finalizing')
        value = self.read()
        self.assertEqual(value['choices'], [11, 2])
        self.assertIn('[################] 10/10 +1/2 reserved', format_progress(value))
        self.assertNotIn('100%', format_progress(value))
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_write_failure_is_best_effort_and_does_not_mask_training_failure(self):
        with patch('experiments.learner.game_progress._write', side_effect=OSError('disk unavailable')), \
             patch('sys.stderr', new=io.StringIO()) as warning:
            self.progress((1, 0))
            self.progress.phase('failed')
        self.assertEqual(warning.getvalue().count('Game progress display unavailable'), 1)
        self.assertEqual(self.progress.value['choices'], [1, 0])

    def test_phase_handoff_and_corrupt_ui_are_never_evidence_gates(self):
        set_phase(self.path, 'verifying')
        self.assertFalse(self.path.exists())
        self.progress((7, 6))
        set_phase(self.path, 'verifying')
        self.assertEqual(self.read()['choices'], [7, 6])
        self.assertEqual(self.read()['phase'], 'verifying')
        self.path.write_text('not-json')
        set_phase(self.path, 'verified')
        output = io.StringIO()
        ProgressDisplay(self.path.parent, stream=output).poll()
        self.assertEqual(output.getvalue(), '')

    def test_calendar_comes_from_delivered_days_and_stays_unknown_until_observed(self):
        self.progress((0, 0))
        self.assertIn('Day ?', format_progress(self.read()))
        self.progress.observe_day(0, 7, max_days=365)
        self.progress((3, 0), 'finalizing')
        self.assertEqual(self.read()['days'], [7, None])
        self.assertIn('Day 7/365', format_progress(self.read()))
        self.progress.observe_day(1, 8, max_days=365)
        self.progress.observe_day(0, 6)  # A stale display update cannot move the shared calendar back.
        self.progress.phase('verifying')
        self.assertEqual(self.read()['choices'], [3, 0])
        self.assertEqual(self.read()['days'], [7, 8])
        self.assertIn('Day 8/365', format_progress(self.read()))
        self.progress.observe_day(1, 366, max_days=365)
        self.progress.phase('verified')
        self.assertIn('Day 366/365', format_progress(self.read()))
        set_phase(self.path, 'verifying')
        self.assertEqual(self.read()['days'], [7, 366])

    def test_malformed_day_metadata_is_ignored_and_legacy_snapshots_still_display(self):
        self.progress.observe_day(0, 12, max_days=365)
        for value in (None, False, 0, -1, '13', 13.5, [], {}):
            self.progress.observe_day(1, value, max_days=value)
        for value in (None, False, -1, 2, '0', 13.5, [], {}):
            self.progress.observe_day(value, 99)
        self.progress((2, 1))
        value = self.read()
        self.assertEqual(value['days'], [12, None])
        self.assertEqual(value['max_days'], 365)
        for name in ('days', 'max_days'):
            value.pop(name)
        self.path.write_text(json.dumps(value))
        output = io.StringIO()
        ProgressDisplay(self.path.parent, stream=output).poll()
        self.assertIn('Day ?', output.getvalue())
        self.assertIn('2/10', output.getvalue())

    def test_deeply_nested_small_sidecar_never_interrupts_handoff_or_cleanup(self):
        from experiments.learner.game_progress import finish_unfinished
        self.path.parent.mkdir()
        nested = '[' * 1500 + '0' + ']' * 1500
        self.assertLess(len(nested), 4096)
        self.path.write_text(nested)
        # The C decoder's nesting threshold varies across Python versions.
        for recursive_decoder in (False, True):
            decoder = (patch('experiments.learner.game_progress.json.loads', side_effect=RecursionError('decoder nesting'))
                       if recursive_decoder else nullcontext())
            with self.subTest(recursive_decoder=recursive_decoder), decoder:
                set_phase(self.path, 'verifying')
                self.assertFalse(finish_unfinished(self.path.parent))
                output = io.StringIO()
                display = ProgressDisplay(self.path.parent, stream=output)
                display.poll()
                self.assertEqual(output.getvalue(), '')
                display.finish(failed=True)
                self.assertIn('Run failed or interrupted', output.getvalue())
        self.assertEqual(self.path.read_text(), nested)

    def test_closed_display_stream_never_masks_poll_finish_or_initialization(self):
        for stage in ('poll', 'finish', 'initialization'):
            with self.subTest(stage=stage):
                self.progress((3, 2), 'verified')
                stream = io.StringIO()
                if stage == 'initialization':
                    stream.close()
                display = ProgressDisplay(self.path.parent, stream=stream)
                if stage == 'finish':
                    display.poll()
                stream.close()
                if stage == 'poll':
                    display.poll()
                display.finish(failed=True)
                self.assertTrue(display.disabled)
                self.assertEqual(self.read()['phase'], 'verified')

    def test_closed_warning_stream_does_not_replace_disk_failure(self):
        stream = io.StringIO(); stream.close()
        with patch('experiments.learner.game_progress._write', side_effect=OSError('disk unavailable')), \
             patch('sys.stderr', new=stream):
            self.progress((1, 0))
            self.progress.phase('failed')
        self.assertTrue(self.progress.warned)

    def test_nonterminal_output_is_throttled_and_phases_force_display(self):
        output = io.StringIO()
        display = ProgressDisplay(self.path.parent, stream=output, clock=lambda: self.now)
        self.progress((1, 0)); display.poll()
        first = output.getvalue()
        self.now = 1
        self.progress((2, 0)); display.poll()
        self.assertEqual(output.getvalue(), first)
        self.progress.phase('verifying'); display.poll()
        self.assertIn('2/10', output.getvalue())
        self.assertIn('verifying', output.getvalue())
        self.assertNotIn('\x1b', output.getvalue())

    def test_terminal_draws_parallel_games_without_claiming_completion_eta(self):
        class Terminal(io.StringIO):
            def isatty(self): return True
        output = Terminal()
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}, clear=True):
            display = ProgressDisplay(self.path.parent, stream=output)
        self.progress((4, 3))
        other = GameProgress(progress_path(self.root, 2), episode=2, ordinary=10, reserved=2)
        other((1, 2))
        display.poll()
        rendered_lines = display.lines
        self.progress.phase('verifying'); display.poll()
        self.assertIn('Bars show choices used, not time remaining.', output.getvalue())
        self.assertIn('Game 1', output.getvalue())
        self.assertIn('Game 2', output.getvalue())
        self.assertIn(f'\x1b[{rendered_lines}F', output.getvalue())
        plain = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output.getvalue())
        self.assertTrue(all(len(line) < 80 for line in plain.splitlines()))

    def test_narrow_terminal_does_not_rewind_wrapped_lines(self):
        class Terminal(io.StringIO):
            def isatty(self): return True
        output = Terminal()
        with patch('shutil.get_terminal_size', return_value=os.terminal_size((60, 24))):
            display = ProgressDisplay(self.path.parent, stream=output)
        self.progress((1, 1)); display.poll()
        self.progress.phase('verifying'); display.poll()
        self.assertNotIn('\x1b', output.getvalue())
        self.assertIn('Verifying game', output.getvalue())
        self.assertTrue(all(len(line) < 60 for line in output.getvalue().splitlines()))

    def test_color_is_terminal_only_and_respects_explicit_opt_out(self):
        class Terminal(io.StringIO):
            def isatty(self): return True
        output = Terminal()
        for environment, enabled in (({'TERM': 'xterm-256color'}, True),
                                      ({'TERM': 'dumb'}, False),
                                      ({'TERM': 'xterm', 'NO_COLOR': ''}, False)):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(terminal_color_enabled(output), enabled)
                self.assertFalse(terminal_color_enabled(io.StringIO()))
        self.assertEqual(color_text('Red', '1;91', False), 'Red')
        self.assertEqual(color_text('Red', '1;91', True), '\x1b[1;91mRed\x1b[0m')
        closed = io.StringIO(); closed.close()
        self.assertFalse(terminal_color_enabled(closed))

    def test_color_keeps_identical_information_and_each_rendered_line_fits(self):
        value = {**self.progress.value, 'choices': [11, 7], 'days': [42, 41], 'max_days': 365}
        for width in (20, 39, 59, 79, 119):
            with self.subTest(width=width):
                plain = terminal_lines(value, width=width)
                colored = terminal_lines(value, width=width, color=True)
                self.assertEqual([re.sub(r'\x1b\[[0-9;]*m', '', line) for line in colored], plain)
                self.assertTrue(all(len(line) <= width for line in plain))
                self.assertIn('42/365', ' '.join(plain))
                self.assertIn('Dialog reserve:', ' '.join(plain))
                self.assertNotIn('%', ' '.join(plain))

    def test_dumb_terminal_is_plain_and_does_not_rewind(self):
        class Terminal(io.StringIO):
            def isatty(self): return True
        output = Terminal()
        with patch.dict(os.environ, {'TERM': 'dumb'}, clear=True):
            display = ProgressDisplay(self.path.parent, stream=output)
            self.progress((1, 1)); display.poll()
            self.progress.phase('verifying'); display.poll()
        self.assertNotIn('\x1b', output.getvalue())
        self.assertIn('Verifying game', output.getvalue())

    def test_cleanup_marks_all_unfinished_slots_and_preserves_verified_state(self):
        phases = ('collecting', 'finalizing', 'awaiting verification', 'verified')
        snapshots = {}
        for episode, phase in enumerate(phases, 1):
            path = progress_path(self.root, episode)
            progress = GameProgress(path, episode=episode, ordinary=10, reserved=2)
            progress((episode, 2), phase)
            snapshots[episode] = path.read_bytes()
        output = io.StringIO()
        display = ProgressDisplay(self.path.parent, stream=output)
        display.poll()
        display.finish(failed=True)
        for episode in (1, 2, 3):
            value = json.loads(progress_path(self.root, episode).read_text())
            self.assertEqual(value['phase'], 'failed')
            self.assertEqual(value['choices'], [episode, 2])
        self.assertEqual(progress_path(self.root, 4).read_bytes(), snapshots[4])
        self.assertEqual(output.getvalue().count('| failed'), 3)
        self.assertIn('Run failed or interrupted', output.getvalue())

    def test_cleanup_error_is_visible_even_if_every_game_was_verified(self):
        self.progress((10, 4), 'verified')
        verified = self.path.read_bytes()
        output = io.StringIO()
        ProgressDisplay(self.path.parent, stream=output).finish(failed=True)
        self.assertEqual(self.path.read_bytes(), verified)
        self.assertIn('| verified', output.getvalue())
        self.assertIn('Run failed or interrupted', output.getvalue())

    def test_real_subprocess_progress_preserves_completion_stdout_and_deadline(self):
        source = """import json, sys, time
from experiments.learner.game_progress import GameProgress
p = GameProgress(sys.argv[1], episode=1, ordinary=10, reserved=2)
p((3, 2)); time.sleep(1.1); p.phase('verifying')
print(json.dumps({'complete': True}))
"""
        process = subprocess.Popen([sys.executable, '-c', source, str(self.path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = io.StringIO()
        try:
            self.assertEqual(ProgressDisplay(self.path.parent, stream=output).wait(process, timeout=5), 0)
            stdout, stderr = process.communicate(timeout=1)
            self.assertEqual(json.loads(stdout), {'complete': True})
            self.assertEqual(stderr, b'')
            self.assertIn('3/10', output.getvalue())
            self.assertIn('verifying', output.getvalue())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                ProgressDisplay(self.path.parent, stream=io.StringIO()).wait(sleeper, timeout=.05)
        finally:
            sleeper.kill(); sleeper.wait(timeout=1)


class CollectionProgressTests(unittest.TestCase):
    def test_day_observer_preserves_full_rollout_rng_and_the_actual_day_limit_boundary(self):
        import torch
        from experiments.learner import contextual as c
        from tests.test_contextual_rollout import initial, request
        from tests.test_selfplay_collection import AlternatingTurns, SelfPlayCollectionTests
        class DayLimit(AlternatingTurns):
            limits = SimpleNamespace(max_days=365)
            def reset(self, *, episode, seed):
                self.pair = tuple(c.PlayerHistory(str(episode), player, schema=c.EXPANDED_SCHEMA) for player in (0, 1))
                for history in self.pair:
                    original = initial(history.player, str(episode), schema=c.EXPANDED_SCHEMA).decision
                    history.emit('decision', day=365, observation=original.observation,
                                 revision=0, offers=original.offers)
                self.counts, self.player = [0, 0], 0
                return request(self.pair[0])
            def step(self, selected):
                current = super().step(selected)
                return self.truncate('game_day_limit') if min(self.counts) else current
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            fixture = SelfPlayCollectionTests()
            model = fixture.model()
            baseline_collector = fixture.collector(model)
            baseline = baseline_collector.run_episode(DayLimit(), episode=1, seed=17,
                max_requests_per_player=2, max_dialog_requests_per_player=1)
            with tempfile.TemporaryDirectory() as directory:
                path = progress_path(directory, 1)
                display = GameProgress(path, episode=1, ordinary=2, reserved=1)
                observed = fixture.collector(model)
                checked = observed.run_episode(DayLimit(), episode=1, seed=17,
                    max_requests_per_player=2, max_dialog_requests_per_player=1, progress=display)
                self.assertEqual(checked.to_dict(), baseline.to_dict())
                self.assertEqual(json.loads(path.read_text())['days'], [366, 366])
                self.assertIn('Day 366/365', format_progress(json.loads(path.read_text())))
                self.assertEqual(json.loads(path.read_text())['choices'], [1, 1])
                for left, right in zip(baseline_collector.sessions, observed.sessions):
                    self.assertTrue(torch.equal(left.generator.get_state(), right.generator.get_state()))
                broken = fixture.collector(model)
                with patch.object(display, 'observe_day', side_effect=RuntimeError('display metadata failure')):
                    recovered = broken.run_episode(DayLimit(), episode=1, seed=17,
                        max_requests_per_player=2, max_dialog_requests_per_player=1, progress=display)
                self.assertEqual(recovered.to_dict(), baseline.to_dict())
                for left, right in zip(baseline_collector.sessions, broken.sessions):
                    self.assertTrue(torch.equal(left.generator.get_state(), right.generator.get_state()))
        finally:
            torch.set_num_threads(threads)

    def test_callback_counts_only_completed_steps_without_changing_rng_or_rollout(self):
        import torch
        from tests.test_selfplay_collection import AlternatingTurns, SelfPlayCollectionTests
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            fixture = SelfPlayCollectionTests()
            model = fixture.model()
            first = fixture.collector(model)
            baseline = first.run_episode(AlternatingTurns(), episode=1, seed=17,
                max_requests_per_player=2, max_dialog_requests_per_player=1)
            second = fixture.collector(model)
            updates = []
            checked = second.run_episode(AlternatingTurns(), episode=1, seed=17,
                max_requests_per_player=2, max_dialog_requests_per_player=1,
                progress=lambda counts, phase: updates.append((counts, phase)))
            self.assertEqual(checked.to_dict(), baseline.to_dict())
            for left, right in zip(first.sessions, second.sessions):
                self.assertTrue(torch.equal(left.generator.get_state(), right.generator.get_state()))
            self.assertEqual(updates, [((0, 0), 'collecting'), ((1, 0), 'collecting'),
                ((1, 1), 'collecting'), ((2, 1), 'collecting'), ((2, 1), 'finalizing')])
            class FailedStep(AlternatingTurns):
                def step(self, selected): raise RuntimeError('actual engine failure')
            updates.clear()
            with self.assertRaisesRegex(RuntimeError, 'actual engine failure'):
                fixture.collector(model).run_episode(FailedStep(), episode=1, seed=17,
                    max_requests_per_player=2, max_dialog_requests_per_player=1,
                    progress=lambda counts, phase: updates.append((counts, phase)))
            self.assertEqual(updates, [((0, 0), 'collecting')])
        finally:
            torch.set_num_threads(original_threads)
