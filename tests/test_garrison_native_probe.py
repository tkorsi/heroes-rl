"""Fixture isolation, branch coverage and diagnostic supervision checks."""
import fcntl
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from experiments.explorer import create_garrison_scenarios as fixtures
from validation.probes import validate_garrison_dialogs as probe
from tests.test_contextual_garrisons import at_garrison
from tests.test_contextual_capabilities import navigate


class GarrisonNativeProbeTests(unittest.TestCase):
    def test_closure_excludes_only_unexecuted_local_controller_and_adds_own_probes(self):
        inherited = probe.capability_implementation_identity()
        actual = probe.implementation_identity()
        removed = "experiments/learner/local_script_training.py"
        self.assertIn(removed, inherited)
        self.assertNotIn(removed, actual)
        self.assertEqual({name: sha for name, sha in actual.items() if name in inherited},
                         {name: sha for name, sha in inherited.items() if name != removed})
        for name in ("engine/ExplorerRL/Capabilities.cpp", "engine/ExplorerRL/ArmyOperations.cpp",
                     "experiments/headless/contextual_replay.py", "experiments/headless/contextual_protocol.py",
                     "experiments/learner/contextual.py", "experiments/learner/contextual_model.py",
                     "validation/probes/validate_garrison_dialogs.py",
                     "experiments/explorer/create_garrison_scenarios.py"):
            self.assertEqual(actual[name], probe.digest(probe.ROOT / name))

    def test_fixtures_preserve_native_outcomes_and_opponent(self):
        original = fixtures.ordinary_payloads()
        for case in fixtures.CASES:
            for player in fixtures.COLORS:
                payloads = fixtures.payloads(case, player)
                for field in ("triggeredEvents", "allowedHeroes", "allowedAbilities", "allowedArtifacts",
                              "allowedSpells", "mapLevels", "heroLevelLimit"):
                    self.assertEqual(payloads["header.json"][field], original["header.json"][field])
                self.assertEqual(payloads["surface_terrain.json"], original["surface_terrain.json"])
                for obj in payloads["objects.json"]:
                    self.assertTrue(obj["instanceName"].rsplit("_", 1)[1].isdigit())
                other = 1 - fixtures.COLORS.index(player)
                for name in (f"town_{other}", f"hero_{other}"):
                    self.assertEqual(next(obj for obj in payloads["objects.json"] if obj["instanceName"] == name),
                                     next(obj for obj in original["objects.json"] if obj["instanceName"] == name))

    def test_locked_occupied_deposit_exposes_different_slot_bug(self):
        for case in ("garrison-locked", "antimagic-locked"):
            value = fixtures.payloads(case, "red")
            hero = next(obj for obj in value["objects.json"] if obj["instanceName"] == "hero_0")
            guard = next(obj for obj in value["objects.json"] if obj["instanceName"] == "diagnostic_0")
            self.assertFalse(guard["options"]["removableUnits"])
            self.assertEqual(hero["options"]["army"][0]["type"], guard["options"]["army"][1]["type"])
            self.assertNotEqual(guard["options"]["army"][0]["type"], hero["options"]["army"][0]["type"])
            self.assertGreater(hero["options"]["army"][0]["amount"], guard["options"]["army"][1]["amount"])

    def test_repeat_and_full_cases_reach_distinct_native_branches(self):
        repeat = fixtures.payloads("mine-repeat", "red")
        mine = repeat["objects.json"][-1]
        self.assertNotIn("owner", mine["options"])
        self.assertEqual(mine["options"]["army"], [])
        full = fixtures.payloads("garrison-full", "blue")
        hero = next(obj for obj in full["objects.json"] if obj["instanceName"] == "hero_1")
        self.assertEqual(len(hero["options"]["army"]), 7)
        self.assertEqual(len(full["objects.json"][-1]["options"]["army"]), 7)

    def test_archives_are_exact_and_preserve_previous_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.vmap", root / "second.vmap"
            manifest = fixtures.create_scenario(first, "mine-repeat", "red")
            fixtures.create_scenario(second, "mine-repeat", "red")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(manifest["diagnostic_only"])
            self.assertFalse(manifest["training_input"])
            self.assertFalse(manifest["engine_verified"])
            self.assertEqual(manifest["optimizer_steps"], 0)
            with self.assertRaises(FileExistsError):
                fixtures.create_scenario(first, "mine-repeat", "red")
            changed = fixtures.payloads("mine-repeat", "red")
            changed["objects.json"][-1]["options"]["owner"] = "red"
            with zipfile.ZipFile(second, "w") as archive:
                for name, value in changed.items():
                    archive.writestr(name, json.dumps(value))
            with self.assertRaises(ValueError):
                fixtures.verify_scenario(second, "mine-repeat", "red")

    def test_all_scheduled_branches_use_unique_fixed_seeds(self):
        schedule = probe.schedule(fixtures.CASES, fixtures.COLORS, 83010000)
        self.assertEqual(len(schedule), 16)
        self.assertEqual(len({item["seed"] for item in schedule}), 16)
        for item in schedule:
            subset = probe.schedule((item["case"],), (item["player"],), 83010000)
            self.assertEqual(subset, [item])

    def test_stack_probe_executes_real_v8_actions_and_preserves_query_history(self):
        # Exercise the same Action constructor, offers and PlayerHistory
        # verifier as the native harness; an invented action name must fail.
        history = at_garrison(object_kind=33, locked=True, stacks={0: (2, 3)})
        parent = history.decision.observation

        class SchemaRun:
            action = probe.CapabilityRun.action

            def __init__(self):
                self.actions, self.checks = [], []

            def own_hero(self):
                return "hero"

            def inspect_request(self):
                return history.decision

            def call(self, selected, label):
                self.actions.append(selected)
                if not history.decision.offers.permits(selected):
                    raise AssertionError("Probe action is not in the delivered v8 support")
                start = len(history.frames)
                if selected.kind == "open_stack":
                    source = next(slot for row in parent.window.armies for slot in row.slots
                                  if slot.ref == selected.selection)
                    panel = probe.c.StackPanel(parent.context, source, (), None, None, None)
                    shown = replace(parent, window=probe.c.Window(
                        probe.c.Context("stack", "hero", "overview", source.ref), None,
                        armies=(), stack_panel=panel, capability_controls=()))
                    navigate(history, selected, shown)
                elif selected.kind == "back":
                    navigate(history, selected, parent, "closed")
                else:
                    raise AssertionError("Stack inspection must return with back, not close the query")
                additions = history.frames[start:]
                result = next(frame.result for frame in additions if frame.kind == "result")
                return None, additions, result

            def check(self, name, status, **proof):
                self.checks.append((name, status, proof))

        run = SchemaRun()
        probe.stack_roundtrip(run)
        self.assertEqual([action.kind for action in run.actions], ["open_stack", "back"])
        self.assertEqual(history.decision.observation, parent)
        self.assertEqual(run.checks[0][2]["native_issued"], 0)
        self.assertIsNotNone(history.active_garrison)
        with self.assertRaisesRegex(ValueError, "Unsupported contextual action"):
            run.action("inspect_stack", subject="hero", selection="own-0")

    def test_official_run_lock_blocks_diagnostic_before_output_or_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lockpath = root / "owner.lock"
            lockpath.touch()
            args = SimpleNamespace(official_lock=lockpath, output=root / "attempt", wall_seconds=60)
            with lockpath.open("r") as owner, patch.object(probe, "guarded_popen") as launch:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    probe.supervise(args, [])
                launch.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_opening_requires_real_owner_event_and_context_order(self):
        hero = SimpleNamespace(subject="hero")
        guard = SimpleNamespace(subject="mine", role="object_garrison")
        frame = SimpleNamespace(seq=9, offers=SimpleNamespace(capability_controls=()))
        target = SimpleNamespace(kind=53, handle="mine")
        event = SimpleNamespace(kind="garrison_requested", subject="hero", partner="mine", object_kind=53, owner="self")
        additions = [SimpleNamespace(event=event, seq=4, kind="event"),
                     SimpleNamespace(event=None, seq=8, kind="context_opened", reason="object_garrison")]
        run = SimpleNamespace(check=lambda *args, **kwargs: None)
        with patch.object(probe, "exchange_rows", return_value=(frame, hero, guard)):
            probe.check_open(run, target, additions, False)
            event.owner = "ally"
            with self.assertRaises(probe.base.ProbeFailure):
                probe.check_open(run, target, additions, False)
            event.owner = "self"
            additions[1].seq = 2
            with self.assertRaises(probe.base.ProbeFailure):
                probe.check_open(run, target, additions, False)


if __name__ == "__main__":
    unittest.main()
