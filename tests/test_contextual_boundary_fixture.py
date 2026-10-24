"""Archive isolation and input validation only; no engine or synthetic outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from validation.probes.create_boundary_fixture import BASELINE, KINDS, ROOT, create_fixture, verify_fixture
from validation.probes.validate_contextual_boundaries import fixture_input


def contents(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_archive(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


class BoundaryFixtureTests(unittest.TestCase):
    def test_townless_removes_only_red_town_and_main_town_field(self):
        original_bytes = BASELINE.read_bytes()
        before = contents(BASELINE)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "townless"
            manifest = create_fixture(output, kind="natural-townless")
            after = contents(output / "natural-townless.vmap")
            self.assertEqual(BASELINE.read_bytes(), original_bytes)
            self.assertEqual((output / "base.vmap").read_bytes(), original_bytes)
            self.assertEqual(before["surface_terrain.json"], after["surface_terrain.json"])
            old_header, new_header = json.loads(before["header.json"]), json.loads(after["header.json"])
            main = old_header["players"]["red"].pop("mainTown")
            self.assertEqual(main, {"x": 4, "y": 7, "l": 0, "generateHero": False})
            self.assertEqual(new_header, old_header)
            old_objects, new_objects = json.loads(before["objects.json"]), json.loads(after["objects.json"])
            self.assertEqual(new_objects, [obj for obj in old_objects if obj["instanceName"] != "town_0"])
            self.assertEqual(sum(obj["type"] == "hero" for obj in new_objects), 2)
            self.assertFalse(manifest["native_validated"])
            self.assertFalse(manifest["natural_outcome_verified"])
            self.assertEqual(manifest["schema"], "heroes-contextual-natural-townless-fixture-v1")
            self.assertNotIn("townless_player", manifest)
            # Compatibility with the already recorded native Red fixture, not an engine outcome assertion.
            self.assertEqual(manifest["fixture"]["sha256"],
                             "e8248b66e5184385f60dddc335691e465b9e18d5fb2aee98a1d034a815f59836")
            self.assertEqual(fixture_input(output / "fixture.json")[0], "natural-townless")

    def test_blue_townless_preserves_all_but_blue_town_and_main_town(self):
        original_bytes = BASELINE.read_bytes()
        before = contents(BASELINE)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "blue-townless"
            manifest = create_fixture(output, kind="natural-townless-blue")
            path = output / "natural-townless-blue.vmap"
            after = contents(path)
            self.assertEqual(BASELINE.read_bytes(), original_bytes)
            self.assertEqual((output / "base.vmap").read_bytes(), original_bytes)
            self.assertEqual(set(before), set(after))
            self.assertEqual(before["surface_terrain.json"], after["surface_terrain.json"])
            old_header, new_header = json.loads(before["header.json"]), json.loads(after["header.json"])
            old_header["players"]["blue"].pop("mainTown")
            self.assertEqual(new_header, old_header)  # Includes Red setup and every victory/fog rule.
            old_objects, new_objects = json.loads(before["objects.json"]), json.loads(after["objects.json"])
            self.assertEqual(new_objects, [obj for obj in old_objects if obj["instanceName"] != "town_1"])
            self.assertEqual([obj for obj in new_objects if obj["type"] == "hero"],
                             [obj for obj in old_objects if obj["type"] == "hero"])
            self.assertEqual(manifest["schema"], "heroes-contextual-natural-townless-blue-fixture-v1")
            self.assertEqual(manifest["townless_player"], "blue")
            self.assertEqual(manifest["evaluator_plan"]["expected_outcomes"], {"red": "win", "blue": "loss"})
            self.assertEqual(manifest["changed_archive_entries"], ["header.json", "objects.json"])
            self.assertEqual([(change["entry"], change["json_pointer"], change["operation"])
                              for change in manifest["changes"]],
                             [("header.json", "/players/blue/mainTown", "remove"),
                              ("objects.json", "/2", "remove")])
            for flag in ("native_validated", "natural_outcome_verified", "engine_launched", "training_performed"):
                self.assertFalse(manifest[flag])
            self.assertEqual(fixture_input(output / "fixture.json")[:2], ("natural-townless-blue", path.resolve()))
            with self.assertRaises(ValueError):
                verify_fixture(path, kind="natural-townless")

    def test_blue_townless_rejects_changed_hero_or_victory_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root / "fixture", kind="natural-townless-blue")
            pristine = contents(root / "fixture/natural-townless-blue.vmap")
            mutations = []
            changed = deepcopy(pristine)
            header = json.loads(changed["header.json"])
            header["triggeredEvents"]["standardDefeat"]["condition"][1]["value"] = 1
            changed["header.json"] = json.dumps(header).encode()
            mutations.append(changed)
            changed = deepcopy(pristine)
            objects = json.loads(changed["objects.json"])
            next(obj for obj in objects if obj["instanceName"] == "hero_1")["options"]["army"][0]["amount"] = 1
            changed["objects.json"] = json.dumps(objects).encode()
            mutations.append(changed)
            for index, entries in enumerate(mutations):
                altered = root / f"unsafe-{index}.vmap"
                write_archive(altered, entries)
                with self.assertRaises(ValueError):
                    verify_fixture(altered, kind="natural-townless-blue")

    def test_den_replaces_only_five_selected_object_fields(self):
        before = contents(BASELINE)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "den"
            manifest = create_fixture(output, kind="unsupported-den")
            after = contents(output / "unsupported-den.vmap")
            self.assertEqual(before["header.json"], after["header.json"])
            self.assertEqual(before["surface_terrain.json"], after["surface_terrain.json"])
            old_objects, new_objects = json.loads(before["objects.json"]), json.loads(after["objects.json"])
            changes = [(i, left, right) for i, (left, right) in enumerate(zip(old_objects, new_objects, strict=True)) if left != right]
            self.assertEqual(len(changes), 1)
            _, left, right = changes[0]
            self.assertEqual(left["instanceName"], "gold_0")
            self.assertEqual(right["instanceName"], "validation_den_0")
            self.assertEqual({key for key in left if left[key] != right[key]}, {"instanceName", "options", "subtype", "template", "type"})
            self.assertEqual([right[key] for key in ("x", "y", "l")], [4, 10, 0])
            self.assertEqual(right["template"]["mask"], ["VV", "BA"])
            self.assertTrue(manifest["footprint_nonoverlapping"])
            self.assertFalse(manifest["native_validated"])
            self.assertEqual(fixture_input(output / "fixture.json")[0], "unsupported-den")

    def test_changed_source_json_type_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = contents(BASELINE)
            header = json.loads(entries["header.json"])
            header["heroLevelLimit"] = True  # Python equality with 1 must not admit changed native JSON type.
            entries["header.json"] = json.dumps(header).encode()
            source = root / "changed.vmap"
            write_archive(source, entries)
            for kind in KINDS:
                output = root / kind
                with self.assertRaises(ValueError): create_fixture(output, kind=kind, baseline=source)
                self.assertFalse(output.exists())

    def test_duplicate_archive_entries_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "duplicate.vmap"
            entries = contents(BASELINE)
            write_archive(source, entries)
            with zipfile.ZipFile(source, "a") as archive:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("objects.json", entries["objects.json"])
            with self.assertRaises(ValueError): create_fixture(root / "out", kind="natural-townless", baseline=source)
            self.assertFalse((root / "out").exists())

    def test_fixture_rejects_extra_field_and_untouched_entry_byte_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root / "fixture", kind="unsupported-den")
            path = root / "fixture/unsupported-den.vmap"
            pristine = contents(path)
            variants = []
            modified = deepcopy(pristine)
            objects = json.loads(modified["objects.json"])
            objects[0]["options"]["owner"] = "blue"
            modified["objects.json"] = json.dumps(objects).encode()
            variants.append(modified)
            modified = deepcopy(pristine)
            modified["header.json"] += b"\n"  # semantic equality alone must not permit undeclared entry edits
            variants.append(modified)
            for index, entries in enumerate(variants):
                altered = root / f"altered-{index}.vmap"
                write_archive(altered, entries)
                with self.assertRaises(ValueError): verify_fixture(altered, kind="unsupported-den")
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError): fixture_input(root / "fixture/fixture.json")

    def test_invalid_kind_existing_output_and_authored_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError): create_fixture(root / "out", kind="unknown")
            self.assertFalse((root / "out").exists())
            root.joinpath("keep.txt").write_text("preserve")
            with self.assertRaises(FileExistsError): create_fixture(root, kind="natural-townless")
            self.assertEqual(root.joinpath("keep.txt").read_text(), "preserve")
        with self.assertRaises(ValueError): create_fixture(ROOT / "validation/boundary-fixture-must-not-exist", kind="unsupported-den")


if __name__ == "__main__":
    unittest.main()
