"""Evidence admission tests; no mocked Heroes rules or performance claims."""
from pathlib import Path
import tempfile
import unittest

from tools.benchmark_native_lifecycle import input_identity, semantic, validate_jobs


class ReplayBenchmarkAdmissionTests(unittest.TestCase):
    def test_capture_and_observation_drift_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'replay.jsonl'
            observation = root / 'observations-p0.jsonl'
            log.write_text('original replay bytes')
            observation.write_text('original observation bytes')
            jobs = [['--log', str(log), '--observations', str(root), '--prefix']]
            validate_jobs(jobs)
            original = input_identity(jobs)
            observation.write_text('altered observation bytes')
            self.assertNotEqual(original, input_identity(jobs))
            added = root / 'observations-p1.jsonl'
            changed = input_identity(jobs)
            added.write_text('added private-player evidence')
            self.assertNotEqual(changed, input_identity(jobs))

    def test_only_timing_metadata_is_ignored_for_equivalence(self):
        base = {'verified': True, 'summary': {'digest': 'abc', 'observations_checked': {'0': 3},
                                            'timing_seconds': {'episode_init': 1}}}
        faster = {'verified': True, 'summary': {'digest': 'abc', 'observations_checked': {'0': 3},
                                              'timing_seconds': {'episode_init': .1}}}
        self.assertEqual(semantic(base), semantic(faster))
        faster['summary']['observations_checked']['0'] = 2
        self.assertNotEqual(semantic(base), semantic(faster))
        faster['summary']['observations_checked']['0'] = 3
        faster['verified'] = False
        self.assertNotEqual(semantic(base), semantic(faster))

    def test_output_or_unknown_options_cannot_enter_input_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'replay.jsonl'
            log.write_text('input')
            for tail in (['--report', str(log)], ['--log', str(log)], ['--experimental-worker'],
                         ['--allow-platform-mismatch'], ['--contextual'], ['--prefix', '--prefix']):
                with self.subTest(tail=tail), self.assertRaises(ValueError):
                    validate_jobs([['--log', str(log), *tail]])


if __name__ == '__main__':
    unittest.main()
