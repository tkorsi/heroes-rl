"""Scenario isolation and complete, reproducible authored-state provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.explorer import create_capability_scenarios as scenarios
from experiments.explorer.create_hero_scenario import hero_scenario_payloads


class CapabilityScenarioTests(unittest.TestCase):
    def test_artifact_fixture_guarantees_extra_item_parking_without_changing_ordinary_map(self):
        ordinary = scenarios.ordinary_payloads()
        for player in scenarios.COLORS:
            value = scenarios.capability_payloads("artifacts", player)
            actor = next(obj for obj in value["objects.json"] if obj["instanceName"] == "hero_" + str(scenarios.COLORS.index(player)))
            self.assertEqual(actor["options"]["artifacts"]["backpack"],
                             ["core:centaurAxe", "core:pendantOfFreeWill"])
            partner = next(obj for obj in value["objects.json"] if obj["instanceName"] == "diagnostic_partner")
            self.assertEqual(partner["options"]["artifacts"], {})
        self.assertEqual(scenarios.ordinary_payloads(), ordinary)
        self.assertEqual(hashlib.sha256((scenarios.MAPS / "hero-management-v2.vmap").read_bytes()).hexdigest(),
                         json.loads((scenarios.MAPS / "hero-management-v2.manifest.json").read_text())["map_sha256"])

    def test_ordinary_revision_changes_only_native_level_limit_and_metadata(self):
        original = hero_scenario_payloads()
        updated = scenarios.ordinary_payloads()
        self.assertEqual(updated["header.json"]["heroLevelLimit"], 0)
        changed = {key for key in updated["header.json"]
                   if updated["header.json"][key] != original["header.json"][key]}
        self.assertEqual(changed, {"name", "description", "heroLevelLimit"})
        self.assertEqual(updated["objects.json"], original["objects.json"])
        self.assertEqual(updated["surface_terrain.json"], original["surface_terrain.json"])
        self.assertEqual(original["header.json"]["heroLevelLimit"], 1)

    def test_fixtures_keep_native_rules_and_no_runtime_variable_grants(self):
        ordinary = scenarios.ordinary_payloads()
        for fixture in scenarios.FIXTURES:
            for player in scenarios.COLORS:
                with self.subTest(fixture=fixture, player=player):
                    value = scenarios.capability_payloads(fixture, player)
                    for field in ("triggeredEvents", "allowedHeroes", "allowedAbilities", "allowedArtifacts",
                                  "allowedSpells", "mapLevels", "heroLevelLimit"):
                        self.assertEqual(value["header.json"][field], ordinary["header.json"][field])
                    self.assertEqual({obj["options"]["owner"] for obj in value["objects.json"] if obj["type"] == "town"},
                                     set(scenarios.COLORS))
                    for obj in value["objects.json"]:
                        self.assertNotIn("variables", obj["options"])
                    if fixture not in ("adventure_spells", "fly_movement", "water_walk_movement"):
                        self.assertEqual(value["surface_terrain.json"], ordinary["surface_terrain.json"])

    def test_movement_spells_are_isolated_and_crossing_has_no_land_detour(self):
        for fixture, spell in (("fly_movement", "fly"), ("water_walk_movement", "waterWalk")):
            for player in scenarios.COLORS:
                value = scenarios.capability_payloads(fixture, player)
                hero = next(obj for obj in value["objects.json"] if obj["type"] == "hero" and obj["options"]["owner"] == player)
                self.assertEqual(hero["options"]["spellBook"], ["core:" + spell])
                self.assertFalse(any(obj["type"] == "boat" for obj in value["objects.json"]))
                for row in (11, 12, 13):
                    self.assertEqual(value["surface_terrain.json"][row], ["wt0_"] * 20)
                self.assertTrue(all(tile != "wt0_" for tile in value["surface_terrain.json"][14]))

    def test_dimension_door_pair_differs_only_by_unseen_landing_occupancy(self):
        for player in scenarios.COLORS:
            clear = scenarios.capability_payloads("dimension_door_clear", player)
            blocked = scenarios.capability_payloads("dimension_door_blocked", player)
            blocked["header.json"]["name"] = clear["header.json"]["name"]
            blocker = blocked["objects.json"].pop()
            self.assertEqual(blocker["instanceName"], "diagnostic_hidden_blocker")
            self.assertEqual(blocker["y"], 18)
            self.assertEqual(clear, blocked)

    def test_initial_full_army_cannot_silently_merge_joining_creature(self):
        for player in scenarios.COLORS:
            value = scenarios.capability_payloads("neutral_overflow", player)
            hero = next(obj for obj in value["objects.json"] if obj["type"] == "hero" and obj["options"]["owner"] == player)
            army = hero["options"]["army"]
            self.assertEqual(len({stack["type"] for stack in army}), 7)
            self.assertNotIn("core:pikeman", {stack["type"] for stack in army})
            neutral = next(obj for obj in value["objects.json"] if obj["instanceName"] == "diagnostic_neutral")
            self.assertEqual(neutral["options"]["character"], "compliant")
            self.assertNotIn("owner", neutral["options"])

    def test_book_purchase_and_spell_fixture_have_distinct_initial_ownership(self):
        for fixture in ("town_controls", "adventure_spells"):
            value = scenarios.capability_payloads(fixture, "blue")
            hero = next(obj for obj in value["objects.json"] if obj["instanceName"] == "hero_1")
            if fixture == "town_controls":
                self.assertEqual(hero["options"]["artifacts"], {})
                self.assertEqual(hero["options"]["spellBook"], [])
            else:
                self.assertEqual(hero["options"]["artifacts"], {"spellbook": "core:spellBook"})
                self.assertEqual(len(hero["options"]["spellBook"]), len(scenarios.ADVENTURE_SPELLS))

    def test_manifest_reconstructs_all_changes_and_archives_are_deterministic(self):
        historical = scenarios.HERO_MAP.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for fixture, player in [(None, None), *[(kind, "red") for kind in scenarios.FIXTURES]]:
                first = directory / (scenarios.scenario_id(fixture, player) + ".vmap")
                second = first.with_name("copy-" + first.name)
                manifest = scenarios.create_scenario(first, fixture=fixture, player=player)
                scenarios.create_scenario(second, fixture=fixture, player=player)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertFalse(manifest["engine_verified"])
                self.assertFalse(manifest["training_performed"])
                reconstructed = deepcopy(hero_scenario_payloads())
                for name, delta in manifest["changed_header_fields"].items():
                    self.assertEqual(reconstructed["header.json"][name], delta["before"])
                    reconstructed["header.json"][name] = delta["after"]
                objects = {obj["instanceName"]: obj for obj in reconstructed["objects.json"]}
                for delta in manifest["object_changes"]:
                    self.assertEqual(objects.get(delta["instance"]), delta["before"])
                    if delta["after"] is None:
                        objects.pop(delta["instance"])
                    else:
                        objects[delta["instance"]] = delta["after"]
                for delta in manifest["terrain_changes"]:
                    x, y = delta["x"], delta["y"]
                    self.assertEqual(reconstructed["surface_terrain.json"][y][x], delta["before"])
                    reconstructed["surface_terrain.json"][y][x] = delta["after"]
                expected = scenarios.verify_scenario(first, fixture=fixture, player=player)
                self.assertEqual(objects, {obj["instanceName"]: obj for obj in expected["objects.json"]})
                self.assertEqual(reconstructed["header.json"], expected["header.json"])
                self.assertEqual(reconstructed["surface_terrain.json"], expected["surface_terrain.json"])
                with self.assertRaises(FileExistsError):
                    scenarios.create_scenario(first, fixture=fixture, player=player)
        self.assertEqual(scenarios.HERO_MAP.read_bytes(), historical)

    def test_verifier_rejects_undeclared_rules_and_boolean_level_limit(self):
        for bad_value in (False, 1, 2):
            altered = scenarios.ordinary_payloads()
            altered["header.json"]["heroLevelLimit"] = bad_value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "changed.vmap"
                with zipfile.ZipFile(path, "w") as archive:
                    for name, value in altered.items():
                        archive.writestr(name, json.dumps(value))
                with self.assertRaisesRegex(ValueError, "exact declared"):
                    scenarios.verify_scenario(path)

    def test_all_committed_maps_and_manifests_match_generator(self):
        for fixture, player in [(None, None), *[(kind, color) for kind in scenarios.FIXTURES for color in scenarios.COLORS]]:
            path = scenarios.MAPS / (scenarios.scenario_id(fixture, player) + ".vmap")
            scenarios.verify_scenario(path, fixture=fixture, player=player)
            metadata = json.loads(path.with_suffix(".manifest.json").read_text())
            self.assertEqual(metadata["map_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(metadata["diagnostic_only"], fixture is not None)


if __name__ == "__main__":
    unittest.main()
