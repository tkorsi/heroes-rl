"""Evaluation caps, fixed checkpoint compatibility and fail-closed scheduling."""
import copy
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.learner import evaluate_builtin_opponent as e


class BuiltinEvaluationTests(unittest.TestCase):
    def test_exact_eight_paired_games_with_model_owned_rng(self):
        cfg = e.config()
        rows = e.schedule(cfg)
        self.assertEqual(len(rows), 8)
        self.assertEqual([r['episode'] for r in rows], list(range(1, 9)))
        for red, blue in zip(rows[::2], rows[1::2]):
            self.assertEqual(red['engine_seed'], blue['engine_seed'])
            self.assertEqual((red['learner_player'], blue['learner_player']), (0, 1))
            self.assertEqual(red['policy_seed'], blue['policy_seed'])
            self.assertNotIn('policy_seeds', red)
        self.assertEqual(len({r['engine_seed'] for r in rows}), 4)

    def test_total_cap_includes_reserved_dialog_choices(self):
        cfg = e.config()
        self.assertEqual(cfg['ordinary_choice_limit'] + cfg['reserved_dialog_choices'], cfg['max_total_choices'])
        for used in (0, 2039, 2040, 2047):
            e.require_choice_budget(used, 200, cfg)
        for used, day in ((2048, 200), (2049, 1), (10, 201)):
            with self.assertRaises(ValueError):
                e.require_choice_budget(used, day, cfg)

    def test_config_cannot_silently_expand_or_coerce_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            for key, bad in (('max_total_choices', 4096), ('max_days', 201),
                             ('shared_wall_seconds', 3601), ('optimizer_steps', False)):
                value = e.config()
                value[key] = bad
                path.write_text(json.dumps(value))
                with patch.object(e, 'CONFIG', path), self.assertRaises(ValueError):
                    e.config()

    def test_source_compatibility_preserves_old_dependencies(self):
        old = {'torch': 'x', 'python': 'y', 'interface': e.INTERFACE,
               'git_commit': 'old', 'source_sha256': {'model.py': 'a', 'lock': 'b'}}
        new = copy.deepcopy(old)
        new.update(git_commit='new')
        new['source_sha256']['evaluator.py'] = 'c'
        e.compatible_sources(old, new)
        for mutate in (lambda v: v['source_sha256'].update({'model.py': 'new'}),
                       lambda v: v.update(interface='old-interface'),
                       lambda v: v['source_sha256'].pop('lock')):
            bad = copy.deepcopy(new)
            mutate(bad)
            with self.assertRaises(ValueError):
                e.compatible_sources(old, bad)

    def test_isolated_failure_advances_without_retry_but_unknown_stops(self):
        common = {'cleanup_verified': True, 'identity_verified': True}
        self.assertTrue(e.can_continue({**common, 'status': 'passed'}))
        self.assertTrue(e.can_continue({**common, 'status': 'native_failure',
                                       'termination_reason': 'unsupported_interface'}))
        for extra in ({'status': 'budget_exhausted'}, {'status': 'incomplete'},
                      {'status': 'native_failure', 'termination_reason': 'protocol_error'},
                      {'status': 'passed', 'cleanup_verified': False},
                      {'status': 'passed', 'identity_verified': False}):
            self.assertFalse(e.can_continue({**common, **extra}))

    def test_all_schedule_rows_exist_before_start_and_are_unscored(self):
        attempts = [e.empty_attempt(row) for row in e.schedule(e.config())]
        self.assertEqual(len(attempts), 8)
        self.assertTrue(all(a['status'] == 'not_started' and a['optimizer_steps'] == 0 for a in attempts))
        self.assertTrue(all(a['rewards'] == {'red': None, 'blue': None} for a in attempts))

    def test_earlier_clock_controls_deadline(self):
        with patch.object(e.time, 'monotonic', return_value=100), patch.object(e.time, 'time', return_value=200):
            self.assertEqual(e.remaining(140, 220), 20)
            self.assertEqual(e.remaining(110, 250), 10)
            self.assertEqual(e.remaining(99, 250), -1)

    def supervisor_fixture(self, root, seconds=10):
        checkpoint = root / 'fixed.pt'
        checkpoint.write_bytes(b'fixture only; never loaded as a model')
        cfg = {**e.config(), 'shared_wall_seconds': seconds,
               'cleanup_reserve_seconds': .3, 'hard_stop_reserve_seconds': .1}
        return {'configuration': cfg, 'execution_output': str(root / 'execution'),
                'schedule': e.schedule(cfg), 'frozen': {}, 'wall_scope': 'harmless supervisor test',
                'checkpoint': {'path': str(checkpoint)}}

    def test_supervisor_advances_fixed_rows_after_preserved_local_failure(self):
        from experiments.headless import process_deadline as guard
        real_spawn = guard.guarded_popen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.supervisor_fixture(root)
            seen = []
            def substitute(command, **kwargs):
                index = int(command[command.index('--index') + 1])
                seen.append(index)
                row = plan['schedule'][index]
                result = {**e.empty_attempt(row), 'status': 'native_failure' if index == 0 else 'passed',
                          'termination_reason': 'unsupported_interface' if index == 0 else 'collection_window',
                          'phase': 'finished', 'cleanup_verified': True, 'identity_verified': True}
                target = root / 'execution' / f"attempt-{row['episode']:02d}.json"
                code = f"from pathlib import Path;Path({str(target)!r}).write_text({json.dumps(result)!r})"
                return real_spawn([sys.executable, '-c', code], **kwargs)
            args = SimpleNamespace(plan=root / 'plan.json', sha256='a' * 64)
            with patch.object(e, 'read_plan', return_value=plan), patch.object(e, 'check_identity'), \
                 patch.object(guard, 'guarded_popen', side_effect=substitute):
                result = e.execute(args)
            self.assertEqual(seen, list(range(8)))
            self.assertEqual(result['status'], 'finished')
            self.assertEqual(result['attempts'][0]['status'], 'native_failure')
            self.assertEqual(result['attempts'][0]['rewards'], {'red': None, 'blue': None})
            self.assertTrue(all(not r.get('supervisor_cleanup_errors') for r in result['attempts']))

    def test_shared_deadline_preserves_incomplete_and_not_started_rows(self):
        from experiments.headless import process_deadline as guard
        real_spawn = guard.guarded_popen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.supervisor_fixture(root, seconds=1.2)
            seen = []
            def stalled(command, **kwargs):
                seen.append(int(command[command.index('--index') + 1]))
                # Real harmless child; never launches VCMI or loads a policy.
                return real_spawn([sys.executable, '-c', 'import time;time.sleep(30)'], **kwargs)
            args = SimpleNamespace(plan=root / 'plan.json', sha256='a' * 64)
            with patch.object(e, 'read_plan', return_value=plan), patch.object(e, 'check_identity'), \
                 patch.object(guard, 'guarded_popen', side_effect=stalled):
                result = e.execute(args)
            self.assertEqual(seen, [0])
            self.assertEqual(result['stop_reason'], 'shared_deadline')
            self.assertEqual(result['attempts'][0]['status'], 'incomplete')
            self.assertTrue(all(r['status'] == 'not_started' and r['reason'] == 'shared_deadline'
                                for r in result['attempts'][1:]))
            self.assertTrue(all(r['rewards'] == {'red': None, 'blue': None} for r in result['attempts']))
            self.assertLess(result['execution_wall_seconds'], 2)
            with self.assertRaises(FileExistsError), patch.object(e, 'read_plan', return_value=plan), \
                 patch.object(e, 'check_identity'):
                e.execute(args)

    def test_tampered_plan_is_rejected_before_importing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                e.read_plan(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()
