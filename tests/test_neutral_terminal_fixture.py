"""Synthetic archive/coverage checks only; no VCMI launch or fabricated outcome."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.learner.contextual import (
    Frame, Hero, ObjectView, Observation, Offers, Tile, VisibleEvent, encode_frame,
)
from validation.probes.create_neutral_terminal_fixture import (
    BASELINE, GUARD_COUNT, ROOT, create_fixture, fixture_input, verify_fixture,
)
from validation.probes.validate_neutral_terminal import classify_neutral_boundary, diagnostic_action


def entries(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def archive(path, values):
    with zipfile.ZipFile(path, "w") as output:
        for name, payload in values.items():
            output.writestr(name, payload)


def view(objects=(), *, heroes=True):
    return Observation((20, 20, 1), tuple(Tile((x, y, 0), 2, 0, False, False)
                       for x, y in ((1, 1), (2, 1), (3, 1))), objects, (0,) * 7,
                       (Hero("own", (1, 1, 0), 1000),) if heroes else (), (), None)


def decision(obs, player=0):
    return Frame("diagnostic", player, 0, 1, "decision", observation=obs,
                 offers=Offers.from_observation(obs), revision=1)


def battle_history(*, day=1):
    values = [Frame("diagnostic", 0, 0, day, "event", event=VisibleEvent("battle_start", side="attacker")),
              Frame("diagnostic", 0, 1, day, "event", event=VisibleEvent("battle_complete", won=False)),
              Frame("diagnostic", 0, 2, day, "event", event=VisibleEvent("hero_lost", subject="own")),
              Frame("diagnostic", 0, 3, day, "terminal", outcome="loss"),
              Frame("diagnostic", 0, 4, day, "view", observation=view(heroes=False))]
    return {"red": [encode_frame(frame) for frame in values], "blue": []}


class NeutralFixtureTests(unittest.TestCase):
    def test_both_colors_change_exact_three_fields_preserve_rules_and_source_bytes(self):
        original_bytes, original = BASELINE.read_bytes(), entries(BASELINE)
        for player, suffix in (("red", "0"), ("blue", "1")):
            with self.subTest(player=player), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "fixture"
                manifest = create_fixture(output, player=player)
                game_map = Path(manifest["fixture"]["path"])
                actual = entries(game_map)
                self.assertEqual(BASELINE.read_bytes(), original_bytes)
                self.assertEqual((output / "base.vmap").read_bytes(), original_bytes)
                expected_header = json.loads(original["header.json"])
                expected_header["players"][player].pop("mainTown")
                self.assertEqual(json.loads(actual["header.json"]), expected_header)
                expected_objects = [obj for obj in json.loads(original["objects.json"])
                                    if obj["instanceName"] != "town_" + suffix]
                next(obj for obj in expected_objects if obj["instanceName"] == "guard_" + suffix)["options"]["amount"] = GUARD_COUNT
                self.assertEqual(json.loads(actual["objects.json"]), expected_objects)
                self.assertEqual(sum(obj["type"] == "hero" for obj in expected_objects), 2)
                for name in original.keys() - {"header.json", "objects.json"}:
                    self.assertEqual(original[name], actual[name])
                self.assertEqual(len(manifest["changes"]), 3)
                self.assertEqual(fixture_input(output / "fixture.json")[:2], (player, game_map))
                for flag in ("native_validated", "engine_launched", "natural_outcome_verified",
                             "neutral_terminal_request_verified", "training_performed"):
                    self.assertFalse(manifest[flag])

    def test_rejects_rule_hero_terrain_and_other_guard_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_fixture(root / "fixture", player="blue")
            original = entries(manifest["fixture"]["path"])
            for mutation in ("rule", "hero", "other_guard", "terrain_bytes"):
                with self.subTest(mutation=mutation):
                    values = deepcopy(original)
                    if mutation == "rule":
                        header = json.loads(values["header.json"])
                        header["triggeredEvents"]["standardDefeat"]["condition"][1]["value"] = 1
                        values["header.json"] = json.dumps(header).encode()
                    elif mutation == "terrain_bytes":
                        values["surface_terrain.json"] += b"\n"
                    else:
                        objects = json.loads(values["objects.json"])
                        selected = next(obj for obj in objects if obj["instanceName"] == ("hero_1" if mutation == "hero" else "guard_0"))
                        if mutation == "hero": selected["options"]["army"][0]["amount"] = 1
                        else: selected["options"]["amount"] = GUARD_COUNT
                        values["objects.json"] = json.dumps(objects).encode()
                    altered = root / f"{mutation}.vmap"
                    archive(altered, values)
                    with self.assertRaises(ValueError): verify_fixture(altered, player="blue")

    def test_manifest_hash_and_change_declarations_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            manifest = create_fixture(root, player="red")
            changed = deepcopy(manifest)
            changed["changes"][-1]["after"] = 99
            (root / "fixture.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError): fixture_input(root / "fixture.json")
            (root / "fixture.json").write_text(json.dumps(manifest))
            target = Path(manifest["fixture"]["path"])
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaises(ValueError): fixture_input(root / "fixture.json")

    def test_rejects_changed_source_invalid_player_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = entries(BASELINE)
            header = json.loads(values["header.json"])
            header["heroLevelLimit"] = True
            values["header.json"] = json.dumps(header).encode()
            source = root / "altered.vmap"
            archive(source, values)
            with self.assertRaises(ValueError): create_fixture(root / "changed", player="red", baseline=source)
            with self.assertRaises(ValueError): create_fixture(root / "invalid", player="neutral")
            with self.assertRaises(FileExistsError): create_fixture(root, player="red")
            with self.assertRaises(ValueError): create_fixture(ROOT / "tests/never-create-fixture", player="red")
            self.assertFalse((root / "changed").exists())
            self.assertFalse((root / "invalid").exists())


class NeutralProbeTests(unittest.TestCase):
    def test_only_current_visible_targets_drive_diagnostic_actions(self):
        gold = ObjectView("gold", (2, 1, 0), 79, 6, "neutral")
        guard = ObjectView("guard", (3, 1, 0), 54, 0, "neutral")
        self.assertIsNone(diagnostic_action(decision(view()), losing_player=0))
        action = diagnostic_action(decision(view((gold,))), losing_player=0)
        self.assertEqual((action.kind, action.x, action.y), ("move_to", 2, 1))
        action = diagnostic_action(decision(view((gold, guard))), losing_player=0)
        self.assertEqual((action.kind, action.x, action.y), ("move_to", 3, 1))
        self.assertEqual(diagnostic_action(decision(view((guard,)), player=1), losing_player=0).kind, "end_turn")
        empty_move = replace(view((guard,)), heroes=(Hero("own", (1, 1, 0), 0),))
        self.assertEqual(diagnostic_action(decision(empty_move), losing_player=0).kind, "end_turn")

    def test_natural_battle_does_not_imply_neutral_request_or_native_replay(self):
        def classify(request, histories=None):
            return classify_neutral_boundary(histories or battle_history(), player="red",
                                              outcomes={"red": "loss", "blue": "win"}, request=request)
        own = classify({"player": 0, "type": 198, "id": 1})
        self.assertTrue(own["natural_battle_loss"])
        self.assertFalse(own["neutral_request_origin"])
        neutral = classify({"player": -1, "type": 198, "id": 1})
        self.assertTrue(neutral["neutral_request_origin"])
        self.assertFalse(neutral["independent_native_replay_verified"])
        self.assertFalse(classify({"player": -1, "type": 90, "id": 1})["neutral_request_origin"])
        self.assertFalse(classify({"player": -1, "type": 198, "id": 1}, battle_history(day=7))["natural_battle_loss"])
        missing = battle_history()
        missing["red"] = [frame for frame in missing["red"] if (frame.get("event") or {}).get("kind") != "battle_complete"]
        self.assertFalse(classify({"player": -1, "type": 198, "id": 1}, missing)["natural_battle_loss"])


if __name__ == "__main__":
    unittest.main()
