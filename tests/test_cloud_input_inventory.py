"""Synthetic inventory only: test files are not game assets or a valid checkpoint."""
import hashlib
import json
import stat
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import inventory_cloud_inputs as i


class CloudInputInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'Data/subdirectory').mkdir(parents=True)
        (self.root / 'Data/test.bin').write_bytes(b'synthetic bytes')

    def test_relative_names_sizes_and_hashes_preserve_inputs(self):
        before = (self.root / 'Data/test.bin').read_bytes()
        self.assertEqual(i.inventory_data(self.root), [{'path':'test.bin','bytes':len(before),
                          'sha256':hashlib.sha256(before).hexdigest()}])
        self.assertEqual((self.root / 'Data/test.bin').read_bytes(), before)

    def test_external_links_and_empty_data_refused(self):
        (self.root / 'Data/link').symlink_to(self.root / 'outside')
        with self.assertRaisesRegex(ValueError, 'symlink'): i.inventory_data(self.root)
        (self.root / 'Data/link').unlink()
        (self.root / 'Data/test.bin').unlink()
        with self.assertRaisesRegex(ValueError, 'empty'): i.inventory_data(self.root)

    def test_wrong_checkpoint_never_creates_manifest(self):
        checkpoint = self.root / 'synthetic.pt'; checkpoint.write_bytes(b'not a checkpoint')
        output = self.root / 'private/manifest.json'
        with self.assertRaisesRegex(ValueError, 'checkpoint size or SHA-256'):
            i.inventory(self.root, checkpoint, output)
        self.assertFalse(output.exists())

    def test_output_inside_checkout_is_refused_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, 'outside the Git checkout'):
            i.inventory(self.root, self.root / 'missing.pt', i.ROOT / 'reports/private-inputs.json')

    def test_private_manifest_is_exclusive_relative_and_not_gameplay_evidence(self):
        checkpoint = self.root / 'synthetic.pt'; checkpoint.write_bytes(b'synthetic fixture')
        expected = i.file_binding(checkpoint)
        output = self.root / 'private/manifest.json'
        with patch.multiple(i, ORIGINAL_BYTES=expected['bytes'], ORIGINAL_SHA256=expected['sha256']):
            result = i.inventory(self.root, checkpoint, output, provenance='Synthetic test only')
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        manifest = json.loads(output.read_text())
        self.assertFalse(manifest['gameplay_performed'])
        self.assertEqual(result['game_data_validation'], 'byte_inventory_only')
        self.assertNotIn(str(self.root), output.read_text())
        before = output.read_bytes()
        with self.assertRaises(FileExistsError): i.inventory(self.root, checkpoint, output)
        self.assertEqual(output.read_bytes(), before)
