"""Archive-only checks; no synthetic battle results or native execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from validation.probes.create_casualty_fixture import BASELINE, ROOT, create_fixture


def read(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


class CasualtyFixtureTests(unittest.TestCase):
    def test_each_guard_preserves_all_other_facts_and_source_bytes(self):
        source = BASELINE.read_bytes()
        baseline = read(BASELINE)
        with tempfile.TemporaryDirectory() as temporary:
            for guard in ("guard_0", "guard_1"):
                output = Path(temporary) / guard
                manifest = create_fixture(output, guard=guard, count=20)
                actual = read(output / "casualty.vmap")
                self.assertEqual(BASELINE.read_bytes(), source)
                self.assertEqual((output / "base.vmap").read_bytes(), source)
                self.assertEqual(actual["header.json"], baseline["header.json"])
                self.assertEqual(actual["surface_terrain.json"], baseline["surface_terrain.json"])
                before, after = json.loads(baseline["objects.json"]), json.loads(actual["objects.json"])
                altered = []
                for left, right in zip(before, after, strict=True):
                    if left != right:
                        altered.append(right["instanceName"])
                        restored = deepcopy(right)
                        self.assertEqual(restored["options"].pop("amount"), 20)
                        expected = deepcopy(left)
                        self.assertEqual(expected["options"].pop("amount"), 6)
                        self.assertEqual(restored, expected)
                self.assertEqual(altered, [guard])
                self.assertFalse(manifest["native_validated"])
                self.assertFalse(manifest["casualties_verified"])
                self.assertEqual(manifest["fixture"]["sha256"], hashlib.sha256((output / "casualty.vmap").read_bytes()).hexdigest())

    def test_reject_changed_rules_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "changed.vmap", Path(temporary) / "output"
            entries = read(BASELINE)
            header = json.loads(entries["header.json"])
            header["heroLevelLimit"] = 2
            entries["header.json"] = json.dumps(header).encode()
            with zipfile.ZipFile(source, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
            with self.assertRaises(ValueError):
                create_fixture(output, baseline=source)
            self.assertFalse(output.exists())

    def test_reject_invalid_count_unknown_subject_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            for kwargs in ({"count": True}, {"count": 6}, {"count": 101}, {"guard": "hero_0"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    create_fixture(output, **kwargs)
                self.assertFalse(output.exists())
            output.mkdir()
            keep = output / "keep.txt"
            keep.write_text("preserve")
            with self.assertRaises(FileExistsError):
                create_fixture(output)
            self.assertEqual(keep.read_text(), "preserve")

    def test_reject_authored_output_path(self):
        with self.assertRaises(ValueError):
            create_fixture(ROOT / "experiments" / "casualty-fixture-must-not-exist")


if __name__ == "__main__":
    unittest.main()
