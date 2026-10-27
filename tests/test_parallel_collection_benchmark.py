"""Offline checks for benchmark isolation, input pinning and honest comparisons."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import benchmark_parallel_collection as bench


class ParallelCollectionBenchmarkTests(unittest.TestCase):
    def test_seed_reservations_are_fresh_and_bound_to_all_eight_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(bench, 'BENCH_ROOT', root):
                first = bench.reserve_schedule(root/'first', 0)
                second = bench.reserve_schedule(root/'second', 1)
            self.assertEqual((first['study'], second['study']), (0, 1))
            a, b = first['schedule'], second['schedule']
            self.assertEqual(len(a), 8)
            self.assertEqual([row['episode'] for row in a], list(range(1, 9)))
            self.assertTrue(all(row['phase'] == 'training' and row['learner_player'] == 0 for row in a))
            self.assertTrue(all(row['learner_player'] == 1 for row in b))
            self.assertFalse({row['engine_seed'] for row in a} & {row['engine_seed'] for row in b})
            self.assertFalse({row['policy_seed'] for row in a} & {row['policy_seed'] for row in b})
            self.assertTrue(all(0 < row['engine_seed'] <= 2**31-1 for row in a+b))

    def test_identical_workload_ratio_requires_every_pair(self):
        attempts = [{'status': 'passed', 'episode_directory': f'/unused/{i}'} for i in range(8)]
        serial = dict(status='passed', attempts=attempts, collection_seconds=80,
                      verification_seconds=40, verified_batch_seconds=120)
        parallel = dict(status='passed', attempts=attempts, collection_seconds=20,
                        verification_seconds=40, verified_batch_seconds=60)
        with mock.patch('tools.acceleration3.campaign_summary.game_digest', side_effect=lambda path: path):
            result = bench.comparison(serial, parallel)
        self.assertTrue(result['all_semantic_matches'])
        self.assertEqual(result['identical_workload_speedup']['collection_seconds'], 4)
        self.assertEqual(result['identical_workload_speedup']['verified_batch_seconds'], 2)
        mismatched = [dict(row) for row in attempts]
        mismatched[4]['episode_directory'] = '/different/game'
        with mock.patch('tools.acceleration3.campaign_summary.game_digest', side_effect=lambda path: path):
            result = bench.comparison(serial, dict(parallel, attempts=mismatched))
        self.assertIsNone(result['identical_workload_speedup'])
        self.assertFalse(result['pairs'][4]['semantic_match'])
        self.assertEqual(result['observed_speedup']['collection_seconds'], 4)

    def test_failed_or_missing_games_never_produce_successful_speedup(self):
        arm = dict(status='failed', attempts=[], collection_seconds=1,
                   verification_seconds=1, verified_batch_seconds=2)
        result = bench.comparison(arm, arm)
        self.assertEqual(len(result['pairs']), 8)
        self.assertTrue(all(row['semantic_match'] is None for row in result['pairs']))
        self.assertFalse(result['both_arms_verified'])
        self.assertEqual(result['observed_speedup'], {})
        self.assertIsNone(result['identical_workload_speedup'])

    def test_copy_preserves_bytes_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'source', root/'copy/input'
            source.write_bytes(b'checkpoint bytes')
            copied = bench.checked_copy(source, target)
            self.assertEqual(copied['sha256'], bench.binding(source)['sha256'])
            with self.assertRaises(FileExistsError):
                bench.checked_copy(source, target)
            self.assertEqual(source.read_bytes(), b'checkpoint bytes')

    def test_isolation_check_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root/'official'
            official.mkdir()
            alias = root/'alias'
            alias.symlink_to(official, target_is_directory=True)
            self.assertTrue(bench.inside(alias/'future', official))
            self.assertFalse(bench.inside(root/'official-other', official))


if __name__ == '__main__':
    unittest.main()
