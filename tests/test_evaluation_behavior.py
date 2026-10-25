"""Offline evidence tests; no engine, policy inference, or optimizer execution."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner import contextual as c
from experiments.learner.evaluation_behavior import analyze_episode, aggregate_episodes
from tests.test_contextual_armies import army
from tests.test_contextual_management import observation


class Frames:
    def __init__(self, player=0):
        self.player, self.frames = player, []

    def add(self, kind, **payload):
        frame = c.Frame("1", self.player, len(self.frames), 1, kind, schema=c.DIALOGS_SCHEMA, **payload)
        self.frames.append(c.encode_frame(frame))
        return frame.seq

    def decision(self, obs):
        return self.add("decision", observation=obs, revision=len(self.frames),
                        offers=c.Offers.from_observation(obs, schema=c.DIALOGS_ACTION_SCHEMA))

    def submit(self, decision, kind, subject=None, **kwargs):
        self.add("submitted", decision_seq=decision,
                 action=c.Action(kind, subject, schema=c.DIALOGS_ACTION_SCHEMA, **kwargs))

    def cutoff(self, obs):
        seq = self.decision(obs)
        self.add("cutoff", decision_seq=seq, reason="collection_window")


def minimal(player=1):
    frames = Frames(player)
    frames.cutoff(observation())
    return frames.frames


def write_episode(folder, red, blue=None, *, status="truncated", tail=b"", complete=True):
    folder = Path(folder)
    manifest = {"history_schema": c.DIALOGS_SCHEMA, "episode": "1", "status": status,
                "termination_reason": "collection_window" if status == "truncated" else "technical_failure",
                "wall_seconds": 2.0, "players": {}}
    for color, frames in (("red", red), ("blue", minimal() if blue is None else blue)):
        path = folder / "agents" / color / "history-v7.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join((json.dumps(f) + "\n").encode() for f in frames) + (tail if color == "red" else b"")
        path.write_bytes(data)
        manifest["players"][color] = {"history_v7": {"path": str(path.relative_to(folder)),
            "sha256": hashlib.sha256(data).hexdigest(), "complete": complete}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class EvaluationBehaviorTests(unittest.TestCase):
    def test_current_movement_support_does_not_reuse_old_enemy_visibility(self):
        frames = Frames()
        base = observation()
        old_sighting = replace(base, tiles=base.tiles + (c.Tile((3, 1, 0), 1, 0, False, False),),
                               objects=(c.ObjectView("enemy-town", (3, 1, 0), 98, 0, "opponent"),))
        frames.add("view", observation=old_sighting)
        seq = frames.decision(base)
        frames.submit(seq, "move_to", "hero", x=3, y=1)
        frames.add("result", decision_seq=seq, result=c.Result("no_route_known", True, (1, 1, 0), 100, False))
        known = replace(base, tiles=base.tiles + (c.Tile((2, 1, 0), 1, 0, False, False),))
        seq = frames.decision(known)
        frames.submit(seq, "move_to", "hero", x=2, y=1)
        moved = replace(known, heroes=(replace(known.heroes[0], position=(2, 1, 0)), known.heroes[1]))
        frames.add("view", observation=moved)
        frames.add("result", decision_seq=seq, result=c.Result("reached", True, (2, 1, 0), 80, True))
        frames.cutoff(moved)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames)
            report = analyze_episode(temp)
        m = report["players"]["red"]["metrics"]
        self.assertEqual(report["status"], "complete")
        self.assertEqual(m["movement"]["results"], {"no_route_known": 1, "reached": 1})
        self.assertEqual((m["movement"]["known_targets"], m["movement"]["unknown_targets"]), (1, 1))
        self.assertEqual(m["movement"]["while_opponent_town_visible"], 0)
        self.assertEqual(m["known_tiles"]["new_after_first_observation"], 1)
        self.assertEqual(m["own_hero_positions"]["hero"], [[1, 1, 0], [2, 1, 0]])
        self.assertEqual(report["throughput"]["observed_choices_per_second"], 1.0)

    def test_cross_army_count_deltas_use_current_own_rows(self):
        rows = (army("first", "hero", "hero", {0: (0, 14)}), army("second", "hero", "other", {0: (0, 7)}))
        before = observation("exchange", armies=rows)
        after = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, 9)}),
                                                 army("second", "hero", "other", {0: (0, 12)})))
        frames = Frames(); seq = frames.decision(before)
        frames.submit(seq, "split_stack", "hero", selection="first-0", destination="second-0", count=5)
        frames.add("view", observation=after)
        frames.add("result", decision_seq=seq, result=c.Result("split", True))
        frames.cutoff(after)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames)
            m = analyze_episode(temp)["players"]["red"]["metrics"]
        effect = m["army_effects"][0]
        self.assertEqual(effect["transfer_scope"], "across_armies")
        self.assertEqual([x["delta"] for x in effect["count_deltas"]], [-5, 5])
        self.assertEqual(effect["observed_cross_army_net_redistribution"], [{"creature": 0, "count": 5}])
        self.assertEqual(m["army_operations"]["observed_cross_army_net_count_redistribution"], 5)

    def test_cross_army_like_stack_swaps_report_net_counts_not_gross_movement(self):
        for right, expected in ((4, 8), (12, 0)):
            with self.subTest(right=right), tempfile.TemporaryDirectory() as temp:
                before = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, 12)}),
                                                           army("second", "hero", "other", {0: (0, right)})))
                after = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, right)}),
                                                          army("second", "hero", "other", {0: (0, 12)})))
                frames = Frames(); seq = frames.decision(before)
                frames.submit(seq, "swap_stacks", "hero", selection="first-0", destination="second-0")
                frames.add("view", observation=after)
                frames.add("result", decision_seq=seq, result=c.Result("swapped", True))
                frames.cutoff(after); write_episode(temp, frames.frames)
                m = analyze_episode(temp)["players"]["red"]["metrics"]
                effect = m["army_effects"][0]
                self.assertEqual(m["army_operations"]["observed_cross_army_net_count_redistribution"], expected)
                self.assertEqual(effect["source_before"]["count"], 12)
                self.assertEqual(effect["destination_before"]["count"], right)
                self.assertEqual(effect["result_status"], "swapped")

    def test_same_creature_swap_has_slot_effects_despite_zero_army_delta(self):
        before = observation("hero", armies=(army("own", "hero", "hero", {0: (0, 12), 1: (0, 4)}),))
        after = observation("hero", armies=(army("own", "hero", "hero", {0: (0, 4), 1: (0, 12)}),))
        frames = Frames(); seq = frames.decision(before)
        frames.submit(seq, "swap_stacks", "hero", selection="own-0", destination="own-1")
        frames.add("view", observation=after)
        frames.add("result", decision_seq=seq, result=c.Result("swapped", True))
        frames.cutoff(after)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames)
            effect = analyze_episode(temp)["players"]["red"]["metrics"]["army_effects"][0]
        self.assertEqual(effect["count_deltas"], [])
        self.assertEqual(len(effect["slot_changes"]), 2)
        self.assertEqual(effect["transfer_scope"], "within_army")

    def test_later_private_panel_is_not_attributed_to_earlier_recruitment(self):
        before = observation("town")
        frames = Frames(); seq = frames.decision(before)
        frames.submit(seq, "recruit", "town", selection="creature", destination="town", count=2)
        frames.add("view", observation=observation())
        frames.add("result", decision_seq=seq, result=c.Result("recruited", True))
        frames.add("view", observation=observation("hero"))
        frames.cutoff(observation())
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames)
            m = analyze_episode(temp)["players"]["red"]["metrics"]
        self.assertEqual(m["recruitment"], {"attempts": 1, "completed": 1})
        self.assertEqual(m["army_effects"][0]["comparison"], "unavailable")
        self.assertIsNone(m["army_effects"][0]["count_deltas"])

    def test_roster_change_is_separate_from_visible_capture_and_hidden_fate(self):
        base = observation(); frames = Frames(); frames.add("view", observation=base)
        extra = c.Town("extra", (3, 2, 0))
        added = replace(base, towns=base.towns + (extra,), heroes=base.heroes[:1])
        frames.add("view", observation=added)
        enemy = c.ObjectView("captured", (3, 1, 0), 98, 0, "opponent")
        visible = replace(added, tiles=added.tiles + (c.Tile((3, 1, 0), 1, 0, False, False),),
                          objects=(enemy,)); frames.add("view", observation=visible)
        frames.add("event", event=c.VisibleEvent("visit", "captured", 98, 0, "opponent"))
        gained = replace(visible, objects=(replace(enemy, owner="self"),), towns=visible.towns + (c.Town("captured", (3, 1, 0)),))
        frames.add("view", observation=gained); frames.cutoff(gained)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames)
            m = analyze_episode(temp)["players"]["red"]["metrics"]
        self.assertEqual(len(m["own_town_roster_changes"]), 2)
        self.assertEqual(len(m["town_capture_evidence"]), 1)
        self.assertEqual(m["town_capture_evidence"][0]["subject"], "captured")
        self.assertEqual(m["own_hero_roster_changes"][0]["removed"], ["other"])
        self.assertNotIn("hero_lost", m["event_counts"])

    def test_failed_bound_prefix_and_missing_seat_are_explicit(self):
        frames = Frames(); seq = frames.decision(observation())
        frames.submit(seq, "move_to", "hero", x=3, y=1)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(temp, frames.frames, status="failed", complete=False, tail=b'{"schema":')
            (Path(temp) / "agents/blue/history-v7.jsonl").unlink()
            report = analyze_episode(temp)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["players"]["red"]["metrics"]["choices"], 1)
        self.assertEqual(report["players"]["red"]["metrics"]["movement"]["results"], {"no_result": 1})
        self.assertIsNone(report["players"]["red"]["metrics"]["movement_attempts"][0]["endpoint_changed"])
        self.assertIn("trailing_partial_json_line", report["players"]["red"]["coverage"]["issues"])
        self.assertIsNone(report["players"]["blue"]["metrics"])
        self.assertEqual(report["aggregate"]["available_player_histories"], 1)
        self.assertEqual(report["aggregate"]["expected_player_histories"], 2)

    def test_hash_tampering_cross_player_and_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); write_episode(root, minimal(0))
            path = root / "agents/red/history-v7.jsonl"; path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "history changed"): analyze_episode(root)
            write_episode(root, minimal(1))
            with self.assertRaisesRegex(ValueError, "player mismatch"): analyze_episode(root)
            manifest = write_episode(root, minimal(0))
            manifest["players"]["red"]["history_v7"]["path"] = "../outside.jsonl"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"): analyze_episode(root)

    def test_missing_manifest_or_binding_never_becomes_zero_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); missing = analyze_episode(root)
            self.assertIsNone(missing["aggregate"])
            manifest = write_episode(root, minimal(0))
            del manifest["players"]["red"]["history_v7"]["sha256"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            report = analyze_episode(root)
            self.assertIsNone(report["players"]["red"]["metrics"])
            other = deepcopy(missing); other["episode_directory"] += "-other"
            aggregate = aggregate_episodes([report, other])
        self.assertEqual(aggregate["aggregate"]["expected_player_histories"], 4)
        self.assertEqual(aggregate["aggregate"]["available_player_histories"], 1)
        self.assertEqual(aggregate["incomplete_episodes"], 2)
        self.assertEqual(aggregate["timed_episodes_with_metrics"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate episode"):
            aggregate_episodes([report, report])

    def test_prior_real_v7_behavior_regression_when_local_evidence_exists(self):
        run = Path(__file__).resolve().parents[1] / "runs/adventure-dialogs-2026-10-25/policy-1"
        paths = [run / f"episode-{i:06d}" for i in (1, 2)]
        if not all((p / "manifest.json").exists() for p in paths):
            self.skipTest("Local retained game captures are intentionally not committed")
        reports = [analyze_episode(p) for p in paths]
        self.assertTrue(all(r["status"] == "complete" for r in reports))
        m = aggregate_episodes(reports)["aggregate"]
        self.assertEqual((m["choices"], m["navigation_and_selection"], m["army_rearrangement"]), (256, 138, 101))
        self.assertEqual(m["movement"]["results"], {"no_route_known": 4, "reached": 2})
        self.assertEqual(m["new_known_tiles_across_player_perspectives"], 22)
        self.assertEqual(m["hiring"]["completed"], 0)
        self.assertEqual(m["recruitment"]["completed"], 1)
        self.assertEqual(m["town_capture_evidence"], 0)
        self.assertEqual(m["player_visible_battle_starts"], 0)


if __name__ == "__main__":
    unittest.main()
