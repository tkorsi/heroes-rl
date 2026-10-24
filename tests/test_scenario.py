"""Scenario checks independent of a live VCMI installation or copyrighted data."""

from collections import deque
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.explorer.create_scenario import create_scenario


class ScenarioTests(unittest.TestCase):
    def _load(self, directory, filename="scenario.vmap"):
        path = Path(directory) / filename
        metadata = create_scenario(path)
        with zipfile.ZipFile(path) as archive:
            payloads = {name: json.loads(archive.read(name)) for name in archive.namelist()}
        return path, metadata, payloads

    def test_reproducible_archive_and_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            first, metadata, _ = self._load(directory, "first.vmap")
            second, second_metadata, _ = self._load(directory, "second.vmap")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(metadata, second_metadata)
            self.assertEqual(metadata["map_sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            with zipfile.ZipFile(first) as archive:
                self.assertTrue(all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist()))

    def test_v3_objects_and_starting_references(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, data = self._load(directory)
        header = data["header.json"]
        objects = data["objects.json"]
        self.assertEqual(header["versionMajor"], 3)
        self.assertIsInstance(objects, list, "VCMI v3 silently requires an object array")
        registry = {obj["instanceName"]: obj for obj in objects}
        self.assertEqual(len(registry), len(objects), "Object instance names must be unique")
        for name, obj in registry.items():
            self.assertRegex(name, r"^.+_[0-9]+$", "VCMI parses a numeric instance-name suffix")
            if obj["type"] == "mine":
                self.assertNotIn("owner", obj["options"], "Neutral ownership uses the serializer default")
        armies = []
        for color, player in header["players"].items():
            hero = registry[player["mainHero"]]
            self.assertEqual(hero["options"]["owner"], color)
            self.assertFalse(player["mainTown"]["generateHero"], "Do not create a second starting hero")
            towns = [obj for obj in objects if obj["type"] == "town" and obj["options"]["owner"] == color]
            self.assertEqual(len(towns), 1)
            self.assertEqual(player["heroes"][player["mainHero"]]["type"], hero["options"]["type"])
            armies.append(hero["options"]["army"])
        self.assertEqual(armies[0], armies[1])
        self.assertEqual(header["triggeredEvents"]["standardVictory"]["condition"], ["standardWin"])
        self.assertEqual(header["triggeredEvents"]["standardDefeat"]["condition"], ["daysWithoutTown", {"value": 7}])

    def test_targets_accessible_and_castle_exit_not_blocked(self):
        """Decode authored footprints and search the traversable surface.

        This catches misplaced anchors, off-map footprints, enclosed targets,
        and a castle spawning its hero in a solid wall. Ordinary encounter
        cells are treated as traversable after their interaction is resolved.
        """
        with tempfile.TemporaryDirectory() as directory:
            _, _, data = self._load(directory)
        level = data["header.json"]["mapLevels"]["surface"]
        width, height = level["width"], level["height"]
        solid = set()
        visits = {}
        for obj in data["objects.json"]:
            mask = obj["template"]["mask"]
            for y, row in enumerate(mask):
                for x, cell in enumerate(row):
                    tile = obj["x"] - len(row) + 1 + x, obj["y"] - len(mask) + 1 + y
                    self.assertTrue(0 <= tile[0] < width and 0 <= tile[1] < height)
                    if cell in "BH":
                        solid.add(tile)
                    if cell in "AT":
                        self.assertNotIn(obj["instanceName"], visits, "Only one visitable tile per object")
                        visits[obj["instanceName"]] = tile
        for color, player in data["header.json"]["players"].items():
            start = visits[player["mainHero"]]
            town = next(obj for obj in data["objects.json"]
                        if obj["type"] == "town" and obj["options"]["owner"] == color)
            self.assertEqual(start, visits[town["instanceName"]])
            self.assertEqual(start, (player["mainTown"]["x"], player["mainTown"]["y"]))
            self.assertNotIn(start, solid)
            queue, reachable = deque([start]), {start}
            while queue:
                x, y = queue.popleft()
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    candidate = x + dx, y + dy
                    if (0 <= candidate[0] < width and 0 <= candidate[1] < height
                            and candidate not in solid and candidate not in reachable):
                        reachable.add(candidate)
                        queue.append(candidate)
            self.assertTrue(set(visits.values()).issubset(reachable))


if __name__ == "__main__":
    unittest.main()
