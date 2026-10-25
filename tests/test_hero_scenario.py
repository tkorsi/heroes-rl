"""Exact versioned scenario delta; no game or policy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.explorer.create_scenario import scenario_payloads
from experiments.explorer.create_hero_scenario import (
    BASELINE, create_hero_scenario, hero_scenario_payloads, verify_hero_scenario,
)


class HeroScenarioTests(unittest.TestCase):
    def test_only_declared_tavern_prohibition_and_metadata_change(self):
        before = scenario_payloads()
        after = hero_scenario_payloads()
        restored = deepcopy(after)
        for key in ("name", "description"):
            restored["header.json"][key] = before["header.json"][key]
        towns = [obj for obj in restored["objects.json"] if obj["type"] == "town"]
        self.assertEqual(len(towns), 2)
        for town in towns:
            self.assertEqual(town["options"]["buildings"]["noneOf"], [])
            self.assertNotIn("core:tavern", town["options"]["buildings"]["allOf"])
            town["options"]["buildings"]["noneOf"] = ["core:tavern"]
        self.assertEqual(restored, before)
        self.assertEqual(after["header.json"]["heroLevelLimit"], 1)

    def test_deterministic_archive_preserves_baseline_and_refuses_mutation(self):
        original = BASELINE.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            left, right = (Path(folder) / name for name in ("left.vmap", "right.vmap"))
            create_hero_scenario(left)
            create_hero_scenario(right)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(BASELINE.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                create_hero_scenario(left)
            payloads = hero_scenario_payloads()
            payloads["header.json"]["heroLevelLimit"] = True
            invalid = Path(folder) / "invalid.vmap"
            with zipfile.ZipFile(invalid, "w") as archive:
                for name, value in payloads.items():
                    archive.writestr(name, json.dumps(value))
            with self.assertRaisesRegex(ValueError, "exact declared variant"):
                verify_hero_scenario(invalid)


if __name__ == "__main__":
    unittest.main()
