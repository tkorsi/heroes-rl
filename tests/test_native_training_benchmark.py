"""A timing result never passes when checkpoint, RNG, capture or source differs."""
from pathlib import Path
import tempfile
import unittest

from tools.benchmark_native_training import outside, summarize


def result(seconds):
    repetition = {name: "same" for name in ('before_full_state_sha256', 'model_sha256', 'gradient_sha256',
        'adam_sha256', 'generator_sha256', 'full_state_sha256', 'state_parts_sha256')}
    repetition.update(seconds=seconds, average_cpu_cores=1.0, sampler_order=[0, 1], report={'samples': 4})
    return {'repetitions': [repetition], 'audit': {'capture': 'same'}, 'parent': {'resume': 'same'},
            'checkout_head': 'same', 'driver': {'sha256': 'same'}}


class NativeBenchmarkTests(unittest.TestCase):
    def test_equal_state_allows_timing_comparison(self):
        summary = summarize({'python': result(4), 'native': result(2)}, ('python', 'native'))
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['kernel_speedup'], 2)
        self.assertTrue(all(summary['exact'].values()))

    def test_any_state_order_capture_or_source_difference_fails(self):
        for key in ('before_full_state_sha256', 'model_sha256', 'gradient_sha256', 'adam_sha256',
                    'generator_sha256', 'full_state_sha256', 'sampler_order', 'report', 'state_parts_sha256',
                    'audit', 'parent', 'checkout_head', 'driver'):
            reference, native = result(4), result(2)
            if key in native:
                native[key] = 'changed'
            else:
                native['repetitions'][0][key] = 'changed'
            with self.subTest(key=key):
                summary = summarize({'python': reference, 'native': native}, ('python', 'native'))
                self.assertEqual(summary['status'], 'numerical_mismatch')
                self.assertFalse(summary['exact'][key])

    def test_incomplete_pair_has_no_speedup(self):
        summary = summarize({'native': result(2)}, ('native', 'python'))
        self.assertEqual(summary['status'], 'incomplete')
        self.assertNotIn('kernel_speedup', summary)

    def test_output_exclusion_resolves_symlinks_and_distinguishes_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'official'
            root.mkdir()
            link = Path(directory) / 'linked'
            link.symlink_to(root, target_is_directory=True)
            self.assertFalse(outside(link / 'diagnostic', root))
            self.assertFalse(outside(root, root))
            self.assertTrue(outside(Path(directory) / 'official-other', root))


if __name__ == '__main__':
    unittest.main()
