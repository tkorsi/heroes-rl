"""Field-by-field fixture isolation; these tests run no engine and assert no outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from validation.probes import create_boundary_fixture as boundary
from validation.probes.create_town_fixtures import (
    BASELINE, FULL_CREATURES, KINDS, MERGE_CREATURES, PLAYERS, ROOT, SCHEMA, UPSTREAM,
    create_fixture, sha, verify_fixture,
)


def contents(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_archive(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


class TownFixtureTests(unittest.TestCase):
    def test_only_chosen_town_army_changes_for_both_kinds_and_players(self):
        source_bytes, before = BASELINE.read_bytes(), contents(BASELINE)
        old_objects = json.loads(before["objects.json"])
        with tempfile.TemporaryDirectory() as temporary:
            for kind in KINDS:
                for player in PLAYERS:
                    with self.subTest(kind=kind, player=player):
                        output = Path(temporary) / f"{kind}-{player}"
                        manifest = create_fixture(output, kind=kind, player=player)
                        target = output / f"{kind}.vmap"
                        after = contents(target)
                        new_objects = json.loads(after["objects.json"])
                        self.assertEqual(set(after), set(before))
                        for name in before:
                            if name != "objects.json":
                                self.assertEqual(after[name], before[name])
                        index = next(i for i, obj in enumerate(old_objects)
                                     if obj["type"] == "town" and obj["options"]["owner"] == player)
                        army = new_objects[index]["options"].pop("army")
                        self.assertEqual(len(army), 7)
                        self.assertEqual(len({stack["type"] for stack in army}), 7)
                        self.assertTrue(all(type(stack["amount"]) is int and stack["amount"] == 1 for stack in army))
                        self.assertEqual("core:pikeman" in {stack["type"] for stack in army}, kind == "capacity-merge")
                        new_objects[index]["options"]["army"] = []
                        self.assertEqual(new_objects, old_objects)  # all hero/tavern/other-town fields included
                        self.assertEqual(BASELINE.read_bytes(), source_bytes)
                        self.assertEqual((output / "base.vmap").read_bytes(), source_bytes)
                        self.assertEqual(manifest["source"]["sha256"], sha(source_bytes))
                        self.assertEqual(manifest["base"]["sha256"], sha(source_bytes))
                        self.assertEqual(manifest["fixture"]["sha256"], sha(target.read_bytes()))
                        self.assertEqual(manifest["schema"], SCHEMA)
                        self.assertEqual(manifest["upstream_commit"], UPSTREAM)
                        self.assertEqual(manifest["changed_archive_entries"], ["objects.json"])
                        self.assertEqual(manifest["changes"], [{"entry": "objects.json",
                            "json_pointer": f"/{index}/options/army", "instance_name": old_objects[index]["instanceName"],
                            "operation": "replace", "before": [], "after": army}])
                        self.assertFalse(manifest["engine_launched"])
                        self.assertFalse(manifest["native_validated"])
                        self.assertFalse(manifest["training_eligible"])
                        self.assertFalse(manifest["random_baseline_outcome_claim"])
                        self.assertIsNone(manifest["evaluator_plan"]["observed_result"])
                        self.assertEqual(verify_fixture(target, kind=kind, player=player)[0], ["objects.json"])

    def test_capacity_pair_differs_only_in_first_slot_type_and_is_deterministic(self):
        self.assertEqual(FULL_CREATURES, ("halberdier", "archer", "marksman", "griffin", "royalGriffin", "swordsman", "crusader"))
        self.assertEqual(MERGE_CREATURES, ("pikeman", *FULL_CREATURES[1:]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in KINDS:
                create_fixture(root / kind, kind=kind)
            full = contents(root / "capacity-full/capacity-full.vmap")
            merged = contents(root / "capacity-merge/capacity-merge.vmap")
            left, right = json.loads(full.pop("objects.json")), json.loads(merged.pop("objects.json"))
            self.assertEqual(full, merged)
            self.assertEqual(left[0]["options"]["army"][0], {"type": "core:halberdier", "amount": 1})
            self.assertEqual(right[0]["options"]["army"][0], {"type": "core:pikeman", "amount": 1})
            left[0]["options"]["army"][0]["type"] = "core:pikeman"
            self.assertEqual(left, right)
            create_fixture(root / "repeat", kind="capacity-full")
            self.assertEqual((root / "repeat/capacity-full.vmap").read_bytes(),
                             (root / "capacity-full/capacity-full.vmap").read_bytes())

    def test_changed_baseline_version_mods_or_native_types_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for field, value in (("versionMajor", 4), ("versionMinor", 1), ("mods", ["hota"]),
                                 ("heroLevelLimit", True)):
                entries = contents(BASELINE)
                header = json.loads(entries["header.json"])
                header[field] = value
                entries["header.json"] = json.dumps(header).encode()
                source, output = root / f"{field}.vmap", root / f"out-{field}"
                write_archive(source, entries)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    create_fixture(output, kind="capacity-full", baseline=source)
                self.assertFalse(output.exists())

    def test_unexpected_hero_or_town_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                entries = contents(BASELINE)
                objects = json.loads(entries["objects.json"])
                objects[index]["options"]["army"] = [{"type": "core:pikeman", "amount": 1}]
                entries["objects.json"] = json.dumps(objects).encode()
                source = root / f"changed-{index}.vmap"
                write_archive(source, entries)
                with self.assertRaises(ValueError):
                    create_fixture(root / f"out-{index}", kind="capacity-merge", baseline=source)
                self.assertFalse((root / f"out-{index}").exists())

    def test_duplicate_archive_and_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = contents(BASELINE)
            duplicate = root / "duplicate.vmap"
            write_archive(duplicate, entries)
            with zipfile.ZipFile(duplicate, "a") as archive, warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("objects.json", entries["objects.json"])
            with self.assertRaises(ValueError):
                create_fixture(root / "duplicate-out", kind="capacity-full", baseline=duplicate)
            entries["header.json"] = b'{"versionMajor":3,' + entries["header.json"].lstrip()[1:]
            source = root / "duplicate-json.vmap"
            write_archive(source, entries)
            with self.assertRaises(ValueError):
                create_fixture(root / "json-out", kind="capacity-full", baseline=source)
            self.assertFalse((root / "duplicate-out").exists())
            self.assertFalse((root / "json-out").exists())

    def test_fixture_verifier_rejects_undeclared_changes_and_wrong_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root / "source", kind="capacity-full")
            target = root / "source/capacity-full.vmap"
            pristine = contents(target)
            with self.assertRaises(ValueError): verify_fixture(target, kind="capacity-merge")
            with self.assertRaises(ValueError): verify_fixture(target, kind="capacity-full", player="blue")
            variants = []
            for path, value in (((0, "options", "army", 0, "amount"), True),
                                ((1, "options", "army", 0, "amount"), 26),
                                ((0, "options", "buildings", "noneOf"), [])):
                entries = deepcopy(pristine)
                objects = json.loads(entries["objects.json"])
                node = objects
                for key in path[:-1]: node = node[key]
                node[path[-1]] = value
                entries["objects.json"] = json.dumps(objects).encode()
                variants.append(entries)
            entries = deepcopy(pristine)
            entries["header.json"] += b"\n"
            variants.append(entries)
            entries = deepcopy(pristine)
            entries["extra.json"] = b"{}"
            variants.append(entries)
            for index, entries in enumerate(variants):
                changed = root / f"tampered-{index}.vmap"
                write_archive(changed, entries)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    verify_fixture(changed, kind="capacity-full")

    def test_changed_engine_pin_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dependencies").mkdir()
            (root / "dependencies/vcmi.json").write_text(json.dumps({"commit": "0" * 40}))
            with patch.object(boundary, "ROOT", root), self.assertRaises(ValueError):
                create_fixture(root / "out", kind="capacity-full")
            self.assertFalse((root / "out").exists())

    def test_invalid_kind_player_authored_path_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kwargs in ({"kind": "low-resource"}, {"kind": "capacity-full", "player": "neutral"}):
                with self.assertRaises(ValueError): create_fixture(root / "out", **kwargs)
                self.assertFalse((root / "out").exists())
            (root / "keep.txt").write_text("preserve")
            with self.assertRaises(FileExistsError): create_fixture(root, kind="capacity-full")
            self.assertEqual((root / "keep.txt").read_text(), "preserve")
        for directory in ("validation", "tools", "vcmi", ".git"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                create_fixture(ROOT / directory / "town-fixture-must-not-exist", kind="capacity-full")


if __name__ == "__main__":
    unittest.main()
