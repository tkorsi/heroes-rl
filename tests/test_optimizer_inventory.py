"""Temporary-file admission tests only; no game assets or simulated gameplay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.native_lifecycle_inventory import SCHEMA, runtime_inventory, require_inventory
from tools import benchmark_native_lifecycle as benchmark


class RuntimeInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / 'Data'
        self.data.mkdir()
        (self.data / 'temporary-fixture.txt').write_text('admission fixture, not game Data')
        self.config = self.root / 'settings.json'
        self.config.write_text('{}')
        self.spec = {'schema': SCHEMA, 'data': [str(self.data)],
                     'configuration': [str(self.config)], 'mods': [], 'additional_libraries': []}

    def test_inventory_detects_same_size_content_add_remove_and_empty_directory(self):
        original = require_inventory(self.spec)
        require_inventory(self.spec, original)
        target = self.data / 'temporary-fixture.txt'
        old = target.read_bytes()
        target.write_bytes(b'x' * len(old))
        with self.assertRaisesRegex(ValueError, 'temporary-fixture.txt'):
            require_inventory(self.spec, original)
        target.write_bytes(old)
        extra = self.data / 'new-empty-directory'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'new-empty-directory'):
            require_inventory(self.spec, original)
        extra.rmdir()
        target.unlink()
        self.assertFalse(runtime_inventory(self.spec)['admitted'])

    def test_collects_all_missing_paths_and_requires_explicit_roles(self):
        self.spec['data'] = [str(self.root / 'absent-data')]
        self.spec['configuration'] = [str(self.root / 'absent-config')]
        result = runtime_inventory(self.spec)
        self.assertFalse(result['admitted'])
        self.assertEqual(len(result['errors']), 2)
        self.spec['data'] = []
        self.assertIn('provide at least one', runtime_inventory(self.spec)['errors'][1])
        del self.spec['mods']
        with self.assertRaises(ValueError):
            runtime_inventory(self.spec)

    def test_file_symlink_target_and_runtime_selectors_are_bound(self):
        other = self.root / 'other'
        other.write_text('same')
        link = self.data / 'linked-file'
        link.symlink_to(other)
        original = require_inventory(self.spec)
        other.write_text('drift')
        with self.assertRaisesRegex(ValueError, 'linked-file'):
            require_inventory(self.spec, original)
        with patch.dict(os.environ, {'XDG_DATA_HOME': '/different-profile'}):
            with self.assertRaisesRegex(ValueError, 'selectors'):
                require_inventory(self.spec, original)
        link.unlink()
        link.symlink_to(self.root, target_is_directory=True)
        self.assertIn('directory symlink unsupported', runtime_inventory(self.spec)['errors'][0])
        self.spec['data'] = [str(link)]
        self.assertIn('canonical real path', runtime_inventory(self.spec)['errors'][0])

    def test_inventory_cli_retains_missing_report_without_launch(self):
        spec = self.root / 'spec.json'
        self.spec['data'] = [str(self.root / 'legally-supplied-Data-required')]
        spec.write_text(json.dumps(self.spec))
        output = self.root / 'inventory-output'
        process = subprocess.run([sys.executable, str(Path(benchmark.__file__)),
                                  '--inventory-only', '--runtime-spec', str(spec),
                                  '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        report = json.loads((output / 'inventory.json').read_text())
        self.assertFalse(report['admitted'])
        self.assertIn('legally-supplied-Data-required', report['errors'][0])

    def test_missing_capture_failure_has_durable_summary_no_native_dispatch(self):
        jobs = self.root / 'jobs.json'
        jobs.write_text(json.dumps([['--log', str(self.root / 'missing-capture')]]))
        output = self.root / 'benchmark-output'
        with patch.object(sys, 'argv', ['benchmark', '--binary', '/not-used', '--jobs', str(jobs),
                                       '--output', str(output)]), patch.object(benchmark, 'measure') as measure:
            with self.assertRaisesRegex(ValueError, 'Missing input'):
                benchmark.main()
            measure.assert_not_called()
        report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['runs'], [])

    def test_runtime_drift_blocks_dispatch_and_retains_actual_inventory(self):
        spec = self.root / 'spec.json'
        spec.write_text(json.dumps(self.spec))
        manifest = self.root / 'inventory.json'
        manifest.write_text(json.dumps(require_inventory(self.spec)))
        self.config.write_text('{"changed":true}')
        log = self.root / 'capture.txt'
        log.write_text('temporary admission fixture')
        jobs = self.root / 'jobs.json'
        jobs.write_text(json.dumps([['--log', str(log)]]))
        output = self.root / 'benchmark-output'
        with patch.object(sys, 'argv', ['benchmark', '--binary', '/bin/true', '--jobs', str(jobs),
                                       '--output', str(output), '--runtime-spec', str(spec),
                                       '--runtime-manifest', str(manifest)]), \
                patch.object(benchmark, 'measure') as measure:
            with self.assertRaisesRegex(ValueError, 'settings.json'):
                benchmark.main()
            measure.assert_not_called()
        report = json.loads((output / 'summary.json').read_text())
        check = report['runtime_admission']['checks'][0]
        self.assertFalse(check['admitted'])
        self.assertIn('runtime', check)
        self.assertEqual(report['status'], 'failed')

    def test_runtime_change_after_measurement_blocks_comparison_acceptance(self):
        spec = self.root / 'spec.json'
        spec.write_text(json.dumps(self.spec))
        manifest = self.root / 'inventory.json'
        manifest.write_text(json.dumps(require_inventory(self.spec)))
        jobs = self.root / 'jobs.json'
        jobs.write_text(json.dumps([['--log', str(self.config)]]))
        output = self.root / 'comparison-output'
        def boundary_only(*args):
            self.config.write_text('{"drift":true}')
            # No native report or gameplay is fabricated; exercise admission
            # immediately after returning across the measurement boundary.
            return {}
        with patch.object(sys, 'argv', ['benchmark', '--binary', '/bin/true', '--jobs', str(jobs),
                                       '--output', str(output), '--runtime-spec', str(spec),
                                       '--runtime-manifest', str(manifest)]), \
                patch.object(benchmark, 'measure', side_effect=boundary_only):
            with self.assertRaisesRegex(ValueError, 'settings.json'):
                benchmark.main()
        report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['runs'], [])
        check = report['runtime_admission']['checks'][-1]
        self.assertEqual(check['stage'], 'before_comparison_acceptance')
        self.assertFalse(check['admitted'])

    def test_pre_dispatch_admission_hook_stops_cold_process(self):
        output = self.root / 'cold'
        with patch.object(benchmark.subprocess, 'Popen') as popen:
            def reject(stage):
                self.assertEqual(stage, 'before_cold_job')
                raise ValueError('temporary fixture drift')
            with self.assertRaisesRegex(ValueError, 'fixture drift'):
                benchmark.measure(Path('/not-launched'), [['--log', '/fixture']], output,
                                  'cold', 1, 1, admission=reject)
            popen.assert_not_called()
        self.assertEqual(json.loads((output / 'cold.json').read_text())['jobs'][0]['status'], 'not_started')

    def test_empty_observation_directory_is_not_admitted(self):
        with self.assertRaisesRegex(ValueError, 'Observation directory needs'):
            benchmark.validate_jobs([['--log', str(self.config), '--observations', str(self.data)]])


if __name__ == '__main__':
    unittest.main()
