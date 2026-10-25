import unittest

from experiments.recording.match_army import ArmyTimelineError, _assemble_timeline


def stack(slot, count):
    return {"slot": slot, "creature": 0, "count": count}


def frame(seq, *, battle=False, absent=False):
    return {"seq": seq, "objects": [] if absent else [[3, 15, 7, 0, 34, 2, 1]],
            "battles": [[15, 7, 0, 0, 1]] if battle else []}


class ArmyTimelineTest(unittest.TestCase):
    def build(self, frames, snapshots):
        return _assemble_timeline(frames, {row["seq"]: (row["seq"] + 2) * 10 for row in frames},
                                  snapshots, player=1, hero_id=3, name="Edric",
                                  initial_slots=[stack(0, 25)])

    def test_snapshot_only_applies_at_its_packet_boundary(self):
        result = self.build([frame(-1), frame(0), frame(1)],
                            [{"packet": 30, "slots": [stack(0, 8), stack(6, 17)]}])
        self.assertEqual(result[-1]["freshness"], "initial_map")
        self.assertEqual(result[0]["slots"], [stack(0, 25)])
        self.assertEqual(result[1]["slots"], [stack(0, 8), stack(6, 17)])
        self.assertEqual(result[1]["source_packet"], 30)

    def test_battle_freezes_entering_army_and_absent_clears_it(self):
        result = self.build([frame(-1), frame(0), frame(1, battle=True),
                             frame(2, battle=True), frame(3, absent=True)],
                            [{"packet": 20, "slots": [stack(0, 8), stack(6, 17)]},
                             {"packet": 40, "slots": [stack(0, 3)]}])
        self.assertEqual(result[1]["status"], "battle")
        self.assertEqual(result[2]["slots"], [stack(0, 8), stack(6, 17)])
        self.assertEqual(result[2]["freshness"], "pre_battle")
        self.assertEqual(result[2]["source_packet"], 20)
        self.assertEqual(result[3]["slots"], [])
        self.assertIsNone(result[3]["position"])
        self.assertEqual(result[3]["status"], "absent")
        self.assertEqual(result[3]["freshness"], "absent")

    def test_survivor_needs_fresh_panel_before_claiming_current_army(self):
        result = self.build([frame(-1), frame(0, battle=True), frame(1), frame(2)],
                            [{"packet": 40, "slots": [stack(0, 3)]}])
        self.assertEqual(result[1]["freshness"], "pre_battle")
        self.assertEqual(result[2]["freshness"], "observed")
        self.assertEqual(result[2]["slots"], [stack(0, 3)])

    def test_panel_from_inside_battle_is_not_a_post_battle_refresh(self):
        result = self.build([frame(-1), frame(0, battle=True), frame(1, battle=True), frame(2)],
                            [{"packet": 30, "slots": [stack(0, 3)]}])
        self.assertEqual(result[2]["freshness"], "pre_battle")
        self.assertEqual(result[2]["slots"], [stack(0, 25)])

    def test_future_observation_never_appears_in_earlier_frame(self):
        result = self.build([frame(-1), frame(0)], [{"packet": 999, "slots": [stack(0, 99)]}])
        self.assertTrue(all(row["slots"] == [stack(0, 25)] for row in result.values()))

    def test_out_of_order_frames_or_snapshots_are_rejected(self):
        with self.assertRaises(ArmyTimelineError):
            self.build([frame(0), frame(-1)], [])
        with self.assertRaises(ArmyTimelineError):
            self.build([frame(-1)], [{"packet": 30, "slots": []}, {"packet": 20, "slots": []}])


if __name__ == "__main__":
    unittest.main()
