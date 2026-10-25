"""Synthetic contract checks, not evidence of live VCMI visibility or outcomes."""
from dataclasses import asdict, replace
import json
import unittest

from experiments.learner.contract import (
    Frame, HistoryCursor, ObjectView, OwnState, Stack, Tile, View, decode_frame,
)


def example_view():
    return View(
        tiles=(Tile((1, 1, 0), 2, False, True), Tile((2, 1, 0), 2, False, False)),
        objects=(ObjectView("local-a", (2, 1, 0), 34, 0, "opponent"),),
        own=OwnState((1, 1, 0), 100, 100, (0,) * 7, (Stack(0, 0, 25),), ((1, 1, 0),), 0),
    )


def decision(seq=0, **kwargs):
    return Frame("test-episode", 0, seq, 1, "decision", example_view(), mask=(True,) * 9, **kwargs)


class ContractTests(unittest.TestCase):
    def test_wire_roundtrip_and_allowlist(self):
        original = decision()
        wire = json.loads(json.dumps(asdict(original)))
        self.assertEqual(decode_frame(wire), original)
        wire["hidden_position"] = [9, 9, 0]
        with self.assertRaises(ValueError):
            decode_frame(wire)

    def test_reject_raw_bridge_and_nested_global_id(self):
        with self.assertRaises(ValueError):
            decode_frame({"type": "observation", "protocol_version": 2})
        wire = json.loads(json.dumps(asdict(decision())))
        wire["view"]["objects"][0]["id"] = 915
        with self.assertRaises(ValueError):
            decode_frame(wire)
        with self.assertRaises(ValueError):
            replace(decision(), schema="bridge-v2")

    def test_hidden_current_positions_and_bad_vocabulary_rejected(self):
        view = example_view()
        with self.assertRaises(ValueError):
            replace(view, objects=(replace(view.objects[0], position=(9, 9, 0)),))
        with self.assertRaises(ValueError):
            replace(view.objects[0], kind=256)
        with self.assertRaises(ValueError):
            replace(view, size=(40, 40, 1))

    def test_history_includes_transient_sighting_without_inventing_fate(self):
        view = example_view()
        empty = replace(view, objects=())
        cursor = HistoryCursor("test-episode", 0)
        frames = [
            Frame("test-episode", 0, 0, 1, "view", empty),
            Frame("test-episode", 0, 1, 1, "view", view),
            Frame("test-episode", 0, 2, 1, "view", empty),
            replace(decision(3), view=empty),
        ]
        for frame in frames:
            cursor.consume(frame)
        self.assertEqual(frames[-1].view.objects, ())
        self.assertEqual(frames[1].view.objects[0].position, (2, 1, 0))
        self.assertEqual(cursor.next_seq, 4)

    def test_holes_duplicates_identity_and_decreasing_time_fail_closed(self):
        cursor = HistoryCursor("test-episode", 0)
        frame = Frame("test-episode", 0, 0, 2, "view", example_view())
        cursor.consume(frame)
        for bad in (frame, replace(frame, seq=2), replace(frame, seq=1, day=1),
                    replace(frame, seq=1, player=1), replace(frame, seq=1, episode="other")):
            with self.assertRaises(ValueError):
                cursor.consume(bad)
        self.assertEqual(cursor.next_seq, 1)

    def test_results_bind_to_exact_decision_and_attempt(self):
        cursor = HistoryCursor("test-episode", 0)
        cursor.consume(decision())
        cursor.consume(Frame("test-episode", 0, 1, 1, "action", action=8, decision_seq=0))
        with self.assertRaises(ValueError):
            cursor.consume(Frame("test-episode", 0, 2, 1, "result", action=8, accepted=True, decision_seq=1))
        cursor.consume(Frame("test-episode", 0, 2, 1, "result", action=8, accepted=True, decision_seq=0))
        cursor.consume(replace(decision(3), day=2))
        self.assertEqual(cursor.next_seq, 4)

    def test_town_only_is_valid_and_does_not_imply_terminal(self):
        view = example_view()
        own = replace(view.own, hero=None, movement=0, movement_max=0, army=())
        frame = replace(decision(), view=replace(view, own=own), mask=(False,) * 8 + (True,))
        HistoryCursor("test-episode", 0).consume(frame)
        with self.assertRaises(ValueError):
            replace(frame, mask=(True,) * 9)

    def test_masked_attempt_and_stale_decision_rejected(self):
        cursor = HistoryCursor("test-episode", 0)
        cursor.consume(replace(decision(), mask=(False,) * 8 + (True,)))
        with self.assertRaises(ValueError):
            cursor.consume(Frame("test-episode", 0, 1, 1, "action", action=0, decision_seq=0))
        with self.assertRaises(ValueError):
            cursor.consume(Frame("test-episode", 0, 1, 1, "view", example_view()))


if __name__ == "__main__":
    unittest.main()
