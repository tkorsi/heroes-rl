"""Synthetic fixture-builder tests; no engine runs or native fog claims.

These tests inspect written archives and rejected inputs. The observations below
are constructed test data, not authenticated VCMI records. Real paired initial
prefix equality is still required before any hidden-world invariance claim.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.learner import contextual as c
from validation.probes.create_contextual_fixtures import BASELINE, ROOT, build_fixtures


def synthetic_frame(player):
    # A fixed sight footprint exercises both the interior and the map border.
    rows = {2: range(2, 7), 3: range(1, 8), 4: range(9),
            **{y: range(10) for y in range(5, 10)},
            10: range(9), 11: range(1, 8), 12: range(2, 7)}
    known = {(x if player == 0 else 19 - x, y, 0)
             for y, columns in rows.items() for x in columns}
    start = (4 if player == 0 else 15, 7, 0)
    observation = c.Observation(
        size=(20, 20, 1),
        tiles=tuple(c.Tile(pos, 0, 0, False, False) for pos in sorted(known)),
        objects=(), resources=(0,) * 7,
        heroes=(c.Hero("synthetic-owned-hero", start, 1560),),
        towns=(c.Town("synthetic-owned-town", start),))
    return c.Frame(episode="synthetic-fixture-test-no-engine", player=player,
                   seq=0, day=1, kind="view", observation=observation)


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        raw = {name: archive.read(name) for name in archive.namelist()}
    return raw, {name: json.loads(data) for name, data in raw.items()}


class ContextualFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="heroes-synthetic-fixtures-")
        self.root = Path(self.temp.name)
        self.source_bytes = BASELINE.read_bytes()
        self.original_raw, self.original = read_archive(BASELINE)

    def tearDown(self):
        try:
            self.assertEqual(BASELINE.read_bytes(), self.source_bytes,
                             "Fixture creation must never modify the production map")
        finally:
            self.temp.cleanup()

    def persist(self, frame, name="observation"):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(c.encode_frame(frame)) + "\n")
        return path

    def reject(self, raw, name="unsafe"):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(raw))
        output = self.root / f"{name}-output"
        with self.assertRaises(ValueError):
            build_fixtures(path, output)
        self.assertFalse(output.exists(), "Rejected input must not leave fixture files")

    def test_both_players_all_modes_preserve_source_known_facts_and_rules(self):
        for player in (0, 1):
            frame = synthetic_frame(player)
            known = {tile.position for tile in frame.observation.tiles}
            reference = self.persist(frame, f"p{player}")
            for variant in ("terrain", "roads", "guard", "all"):
                with self.subTest(player=player, variant=variant):
                    output = self.root / f"p{player}-{variant}"
                    report = build_fixtures(reference, output, variant=variant)
                    raw, actual = read_archive(output / "variant.vmap")
                    self.assertEqual((output / "base.vmap").read_bytes(), self.source_bytes)
                    self.assertEqual(raw["header.json"], self.original_raw["header.json"])
                    self.assertEqual(actual["objects.json"][:len(self.original["objects.json"])],
                                     self.original["objects.json"])
                    for x, y, _ in known:
                        self.assertEqual(actual["surface_terrain.json"][y][x],
                                         self.original["surface_terrain.json"][y][x])
                    changed = {(x, y, 0) for y in range(20) for x in range(20)
                               if actual["surface_terrain.json"][y][x]
                               != self.original["surface_terrain.json"][y][x]}
                    self.assertFalse(changed & known)
                    self.assertEqual(len(changed), {"guard": 0, "terrain": 2, "roads": 2, "all": 4}[variant])
                    self.assertEqual(changed, {tuple(item["position"]) for item in report["changes"]
                                               if "position" in item})
                    self.assertEqual(set(raw), set(self.original_raw))
                    for entry in set(raw) - set(report["changed_archive_entries"]):
                        self.assertEqual(raw[entry], self.original_raw[entry])

    def test_guard_full_native_mask_is_hidden_and_clear_for_both_players(self):
        # Every baseline template cell is occupied. Bounding rectangles therefore
        # provide an independent collision check for this fixed authored map.
        occupied = set()
        for obj in self.original["objects.json"]:
            mask = obj["template"]["mask"]
            self.assertTrue(all(set(row) <= set("VBA") for row in mask))
            occupied.update((x, y, obj["l"])
                            for x in range(obj["x"] - len(mask[0]) + 1, obj["x"] + 1)
                            for y in range(obj["y"] - len(mask) + 1, obj["y"] + 1))
        original_guard = next(obj for obj in self.original["objects.json"] if obj["instanceName"] == "guard_0")
        for player in (0, 1):
            with self.subTest(player=player):
                frame = synthetic_frame(player)
                known = {tile.position for tile in frame.observation.tiles}
                output = self.root / f"guard-{player}"
                report = build_fixtures(self.persist(frame, f"guard-{player}"), output, variant="guard")
                _, actual = read_archive(output / "variant.vmap")
                self.assertEqual(len(actual["objects.json"]), len(self.original["objects.json"]) + 1)
                guard = actual["objects.json"][-1]
                self.assertEqual(guard["template"]["mask"], ["VV", "VA"])
                x, y, z = guard["x"], guard["y"], guard["l"]
                footprint = {(x, y, z), (x - 1, y, z), (x, y - 1, z), (x - 1, y - 1, z)}
                self.assertTrue(all(0 <= px < 20 and 0 <= py < 20 and pz == 0 for px, py, pz in footprint))
                self.assertFalse(footprint & (known | occupied))
                self.assertEqual(guard["options"], original_guard["options"])
                self.assertNotIn("owner", guard["options"])
                self.assertEqual(guard["template"], original_guard["template"])
                self.assertEqual(guard["type"], "monster")
                change = report["changes"][0]
                self.assertEqual({tuple(pos) for pos in change["occupied_footprint"]}, footprint)
                self.assertEqual({tuple(pos) for pos in change["visible_footprint"]}, footprint)
                adjacent = {tuple(pos) for pos in change["known_adjacent_clear_tiles"]}
                self.assertTrue(adjacent and adjacent <= known - occupied)
                self.assertTrue(all(max(abs(px - x), abs(py - y)) == 1 for px, py, _ in adjacent))

    def test_manifest_binds_file_bytes_and_does_not_claim_native_validation(self):
        reference = self.persist(synthetic_frame(0))
        output = self.root / "manifest"
        report = build_fixtures(reference, output)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), report)
        for key in ("source_map", "base", "variant_map", "observation_reference"):
            self.assertEqual(report[key]["sha256"], hashlib.sha256(Path(report[key]["path"]).read_bytes()).hexdigest())
        self.assertFalse(report["native_validated"] or report["engine_launched"] or report["training_performed"])
        self.assertIn("never training data", report["scope"])
        self.assertIn("does not prove", report["observation_reference"]["authenticity"])
        self.assertTrue(any("prefix" in requirement for requirement in report["required_native_checks"]))

    def test_identical_input_produces_identical_variant_archives(self):
        for player in (0, 1):
            with self.subTest(player=player):
                reference = self.persist(synthetic_frame(player), f"repeat-{player}")
                first = self.root / f"first-{player}"
                second = self.root / f"second-{player}"
                build_fixtures(reference, first)
                build_fixtures(reference, second)
                self.assertEqual((first / "variant.vmap").read_bytes(), (second / "variant.vmap").read_bytes())

    def test_existing_output_is_rejected_without_overwriting_files(self):
        output = self.root / "occupied"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"existing user data")
        with self.assertRaises(FileExistsError):
            build_fixtures(self.persist(synthetic_frame(0)), output)
        self.assertEqual(sentinel.read_bytes(), b"existing user data")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_authored_source_directories_are_rejected(self):
        reference = self.persist(synthetic_frame(0))
        for dirname in ("engine", "experiments", "dependencies"):
            with self.subTest(dirname=dirname):
                output = ROOT / dirname / self.root.name
                self.assertFalse(output.exists())
                with self.assertRaises(ValueError):
                    build_fixtures(reference, output)
                self.assertFalse(output.exists())

    def test_noninitial_or_window_observations_are_rejected(self):
        for player in (0, 1):
            frame = synthetic_frame(player)
            panel = c.Window(c.Context("hero", "synthetic-owned-hero"),
                             (c.Stack(0, 0, 25),), primary_skills=(1, 1, 1, 1), mana=10, level=1)
            variants = {
                "later-day": replace(frame, day=2),
                "moved-hero": replace(frame, observation=replace(frame.observation,
                    heroes=(c.Hero("synthetic-owned-hero", (3, 7, 0), 1000),))),
                "open-window": replace(frame, observation=replace(frame.observation, window=panel)),
            }
            for name, modified in variants.items():
                with self.subTest(player=player, case=name):
                    self.reject(c.encode_frame(modified), f"p{player}-{name}")

    def test_missing_identity_old_schema_and_duplicate_keys_are_rejected(self):
        raw = c.encode_frame(synthetic_frame(0))
        for field in ("schema", "player", "episode"):
            modified = deepcopy(raw)
            del modified[field]
            with self.subTest(field=field):
                self.reject(modified, f"missing-{field}")
        self.reject({**raw, "schema": "heroes-player-history-v1"}, "old-schema")
        self.reject(raw["observation"], "bare-observation")
        path = self.root / "duplicate.json"
        path.write_text('{"player": 0, ' + json.dumps(raw)[1:])
        with self.assertRaises(ValueError):
            build_fixtures(path, self.root / "duplicate-output")
        self.assertFalse((self.root / "duplicate-output").exists())

    def test_reference_terrain_or_roads_inconsistent_with_baseline_are_rejected(self):
        frame = synthetic_frame(0)
        for field in ("terrain", "road"):
            tile = replace(frame.observation.tiles[0], **{field: 1})
            observation = replace(frame.observation, tiles=(tile,) + frame.observation.tiles[1:])
            with self.subTest(field=field):
                self.reject(c.encode_frame(replace(frame, observation=observation)), f"changed-{field}")

    def test_no_hidden_space_is_rejected_for_both_players(self):
        for player in (0, 1):
            frame = synthetic_frame(player)
            observation = replace(frame.observation, tiles=tuple(
                c.Tile((x, y, 0), 0, 0, False, False) for y in range(20) for x in range(20)))
            with self.subTest(player=player):
                self.reject(c.encode_frame(replace(frame, observation=observation)), f"all-known-{player}")

    def test_modified_baseline_rules_or_hidden_terrain_are_rejected(self):
        for field in ("header.json", "surface_terrain.json"):
            with self.subTest(entry=field):
                altered = deepcopy(self.original)
                if field == "header.json":
                    altered[field]["triggeredEvents"] = {}
                else:
                    altered[field][0][0] = "wt20_"
                baseline = self.root / f"modified-{field}.vmap"
                with zipfile.ZipFile(baseline, "w") as archive:
                    for name, value in altered.items():
                        archive.writestr(name, json.dumps(value))
                before = baseline.read_bytes()
                output = self.root / f"modified-{field}-output"
                with self.assertRaises(ValueError):
                    build_fixtures(self.persist(synthetic_frame(0)), output, baseline_map=baseline)
                self.assertFalse(output.exists())
                self.assertEqual(baseline.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
