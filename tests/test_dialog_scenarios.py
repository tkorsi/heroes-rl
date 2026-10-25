"""Exact diagnostic starting-state provenance; no engine or learning process."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.explorer import create_dialog_scenarios as p
from experiments.explorer.create_hero_scenario import hero_scenario_payloads


class DialogScenarioTests(unittest.TestCase):
    def test_all_variants_keep_normal_rules_terrain_and_other_player(self):
        original = hero_scenario_payloads()
        for fixture in p.FIXTURES:
            for color in p.COLORS:
                with self.subTest(fixture=fixture, color=color):
                    value = p.dialog_scenario_payloads(fixture, color)
                    self.assertEqual(value["surface_terrain.json"], original["surface_terrain.json"])
                    for field in ("triggeredEvents", "heroLevelLimit", "allowedHeroes", "allowedAbilities",
                                  "allowedArtifacts", "allowedSpells", "mapLevels"):
                        self.assertEqual(value["header.json"][field], original["header.json"][field])
                    other = "blue" if color == "red" else "red"
                    self.assertEqual(value["header.json"]["players"][other], original["header.json"]["players"][other])
                    previous = {obj["instanceName"]: obj for obj in original["objects.json"]}
                    for obj in value["objects.json"]:
                        if obj["type"] == "town" or (obj["type"] == "hero" and obj["options"]["owner"] == other):
                            self.assertEqual(obj, previous[obj["instanceName"]])
                        self.assertNotIn("variables", obj["options"])
                        self.assertNotIn("joinOnlyForMoney", obj["options"])

    def test_native_hero_defaults_without_fabricated_skills_or_books(self):
        for fixture, names in (("neutral_paid", {"core:adela"}), ("neutral_paid_unaffordable", {"core:adela"}), ("scholar", {"core:thane"}),
                               ("scholar_exchange", {"core:thane", "core:josephine"})):
            value = p.dialog_scenario_payloads(fixture, "blue")
            heroes = [obj for obj in value["objects.json"] if obj["type"] == "hero" and obj["options"]["owner"] == "blue"]
            self.assertEqual({obj["options"]["type"] for obj in heroes}, names)
            for hero in heroes:
                for field in ("primarySkills", "secondarySkills", "spellBook", "artifacts"):
                    self.assertNotIn(field, hero["options"])

    def test_refusal_attack_is_only_the_native_never_flees_fixture_delta(self):
        ordinary = p.dialog_scenario_payloads("neutral_free", "red")
        attack = p.dialog_scenario_payloads("neutral_free_attack", "red")
        ordinary["header.json"]["name"] = attack["header.json"]["name"]
        monster = next(obj for obj in ordinary["objects.json"] if obj["type"] == "monster")
        self.assertFalse(monster["options"]["neverFlees"])
        monster["options"]["neverFlees"] = True
        self.assertEqual(ordinary, attack)

    def test_unaffordable_fixture_changes_only_declared_initial_army_counts(self):
        for color in p.COLORS:
            ordinary = p.dialog_scenario_payloads("neutral_paid", color)
            unaffordable = p.dialog_scenario_payloads("neutral_paid_unaffordable", color)
            ordinary["header.json"]["name"] = unaffordable["header.json"]["name"]
            hero = next(obj for obj in ordinary["objects.json"] if obj["type"] == "hero" and obj["options"]["owner"] == color)
            monster = next(obj for obj in ordinary["objects.json"] if obj["type"] == "monster")
            self.assertEqual(hero["options"]["army"], [{"type":"core:pikeman", "amount":25}])
            self.assertEqual(monster["options"]["amount"], 6)
            hero["options"]["army"][0]["amount"] = 1500
            monster["options"]["amount"] = 180
            self.assertEqual(ordinary, unaffordable)

    def test_deterministic_archive_and_exact_declared_delta(self):
        previous = {path: path.read_bytes() for path in (p.BASELINE, p.HERO_MAP)}
        with tempfile.TemporaryDirectory() as folder:
            left, right = (Path(folder) / name for name in ("left.vmap", "right.vmap"))
            manifest = p.create_dialog_scenario(left, "school_war", "red")
            p.create_dialog_scenario(right, "school_war", "red")
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertFalse(manifest["training_input"])
            self.assertFalse(manifest["engine_verified"])
            original = hero_scenario_payloads()
            changed = {obj["instanceName"]: obj for obj in original["objects.json"]}
            for delta in manifest["object_changes"]:
                self.assertEqual(changed.get(delta["instance"]), delta["before"])
                if delta["after"] is None: changed.pop(delta["instance"])
                else: changed[delta["instance"]] = delta["after"]
            expected = p.dialog_scenario_payloads("school_war", "red")
            self.assertEqual(changed, {obj["instanceName"]: obj for obj in expected["objects.json"]})
            with self.assertRaises(FileExistsError): p.create_dialog_scenario(left, "school_war", "red")
            corrupt = deepcopy(expected); corrupt["header.json"]["heroLevelLimit"] = True
            invalid = Path(folder) / "invalid.vmap"
            with zipfile.ZipFile(invalid, "w") as archive:
                for name, data in corrupt.items(): archive.writestr(name, json.dumps(data))
            with self.assertRaisesRegex(ValueError, "exact declared"):
                p.verify_dialog_scenario(invalid, "school_war", "red")
        for path, contents in previous.items(): self.assertEqual(path.read_bytes(), contents)

    def test_committed_fixture_bytes_match_exact_generator(self):
        for fixture in p.FIXTURES:
            for color in p.COLORS:
                path = p.MAPS / (p.scenario_id(fixture, color) + ".vmap")
                if not path.exists():
                    self.fail("Generate and track every declared diagnostic map before source freeze")
                p.verify_dialog_scenario(path, fixture, color)
                metadata = json.loads(path.with_suffix(".manifest.json").read_text())
                self.assertEqual(metadata["map_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__": unittest.main()
