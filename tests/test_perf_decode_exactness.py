"""Shared tile decoding and single observation decoding must equal the reference decoders."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from experiments.learner import contextual as c
from tests.test_contextual_armies import history_at_armies
from tests.test_contextual_dialogs import history_at

VALID = {"position": [1, 2, 0], "terrain": 3, "road": 0, "blocked": False, "visitable": True}


def reference_tile(raw):
    return c._decode_simple(raw, c.Tile, ("position",))


def outcome(function, raw):
    """The value (by repr, which tells bool from int) or the exact error of decoding raw."""
    try:
        return "value", repr(function(raw))
    except ValueError as error:
        return "error", str(error)


def raw_tiles():
    variants = [VALID, {**VALID, "blocked": True, "visitable": False}, {**VALID, "position": [0, 0, 0], "road": 255}]
    for key, values in (("position", ([1, 2], [1, 2, 0, 0], [True, 2, 0], [1.0, 2, 0], (1, 2, 0), [-1, 2, 0], "x")),
                        ("terrain", (True, 3.0, -1, 256, "3", None)),
                        ("road", (False, 0.0, 300)),
                        ("blocked", (0, 1, None, "false")),
                        ("visitable", (1, 0.0))):
        variants.extend({**VALID, key: value} for value in values)
    variants += [{key: value for key, value in VALID.items() if key != "road"}, {**VALID, "extra": 1}, [], None, "tile"]
    return variants


class TileDecodingTests(unittest.TestCase):
    def setUp(self):
        c._TILE_CACHE.clear()
        c._LAST_TILES[:] = None, None

    def test_shared_tiles_decode_exactly_as_the_reference_or_fail_identically(self):
        for _ in range(2):  # cold, then every valid tile is shared
            for raw in raw_tiles():
                with self.subTest(raw=raw):
                    self.assertEqual(outcome(c._decode_tile, deepcopy(raw)), outcome(reference_tile, deepcopy(raw)))
        shared = c._decode_tile(deepcopy(VALID))
        self.assertIs(c._decode_tile(deepcopy(VALID)), shared)
        self.assertEqual(tuple(map(type, shared.position)), (int, int, int))

    def test_equal_values_of_other_types_never_share_a_tile(self):
        c._decode_tile(deepcopy(VALID))
        for key, value in (("blocked", 0), ("visitable", 1), ("terrain", 3.0), ("position", [1, 2, False])):
            with self.subTest(key=key):
                raw = {**VALID, key: value}
                self.assertEqual(raw, VALID)  # equal under ==, yet another decoding
                self.assertEqual(outcome(c._decode_tile, raw), outcome(reference_tile, raw))
                self.assertEqual(outcome(c._decode_tile, raw)[0], "error")

    def test_bounded_sharing_stays_exact(self):
        with patch.object(c, "_TILE_CACHE_LIMIT", 2):
            for x in range(7):
                raw = {**VALID, "position": [x % 3, 0, 0]}
                self.assertEqual(outcome(c._decode_tile, raw), outcome(reference_tile, raw))
                self.assertLessEqual(len(c._TILE_CACHE), 2)

    def test_identical_tile_lists_reuse_their_tiles_and_any_type_change_decodes_again(self):
        raw = [{**VALID, "position": [x, 0, 0]} for x in range(4)]
        first = c._decode_tiles(deepcopy(raw))
        self.assertEqual(first, tuple(reference_tile(item) for item in deepcopy(raw)))
        self.assertIs(c._decode_tiles(deepcopy(raw)), first)
        changed = deepcopy(raw)
        changed[2]["blocked"] = 0
        self.assertEqual(changed, raw)
        with self.assertRaisesRegex(ValueError, "boolean"):
            c._decode_tiles(changed)
        self.assertIs(c._decode_tiles(deepcopy(raw)), first)  # a failure never replaces the reusable list
        moved = deepcopy(raw)
        moved[0]["position"][0] = 9
        self.assertEqual(c._decode_tiles(moved)[0].position, (9, 0, 0))
        for bad in ([object()], ({**VALID},), "tiles"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c._decode_tiles(bad)


class ObservationDecodingTests(unittest.TestCase):
    def frames(self):
        for history in (history_at(), history_at_armies(), history_at_armies(player=1)):
            yield from history.frames

    def test_observations_decode_to_their_original_with_cold_and_warm_caches(self):
        for _ in range(2):
            for frame in self.frames():
                if frame.observation is None:
                    continue
                raw = c.encode_observation(frame.observation, schema=frame.schema)
                decoded = c.decode_observation(json.loads(json.dumps(raw)), schema=frame.schema)
                self.assertEqual(decoded, frame.observation)
                self.assertEqual(repr(decoded), repr(frame.observation))

    def test_a_supplied_observation_gives_exactly_the_decoded_frame(self):
        frames = list(self.frames())
        some = next(frame.observation for frame in frames if frame.observation is not None)
        for frame in frames:
            raw = json.loads(c.canonical_frame(frame))
            if raw.get("observation") is None:
                with self.assertRaises(ValueError):
                    c.decode_frame(deepcopy(raw), observation=some)
                continue
            observation = c.decode_observation(deepcopy(raw["observation"]), schema=raw["schema"])
            supplied, decoded = c.decode_frame(deepcopy(raw), observation=observation), c.decode_frame(deepcopy(raw))
            self.assertEqual(supplied, decoded)
            self.assertEqual(repr(supplied), repr(decoded))
            self.assertEqual(c.canonical_frame(supplied), c.canonical_frame(frame))

    def test_frame_count_is_the_length_of_the_frames(self):
        for history in (history_at(), history_at_armies()):
            self.assertEqual(history.frame_count, len(history.frames))
            self.assertTrue(history)  # an object without __len__ stays truthy


if __name__ == "__main__":
    unittest.main()
