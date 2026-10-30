"""Per-episode verification shortcuts must produce exactly the reference bytes and errors."""
from collections import namedtuple
from dataclasses import dataclass, fields, replace
import enum
import json
import unittest

from experiments.headless import contextual_evidence, contextual_protocol
from experiments.headless import contextual_replay_history as replay_history
from experiments.learner import contextual as c
from tests.test_builtin_v8_cutoff import mandatory_history, spell_target_history
from tests.test_contextual_armies import history_at_armies
from tests.test_contextual_capabilities import market_history
from tests.test_contextual_dialogs import history_at
from tests.test_contextual_necromancy import history_with_necromancy
from tests.test_contextual_model import map_history
from tests.test_passive_army_information import informed_observation


def reference_wire(frame):
    return json.dumps(c.encode_frame(frame), sort_keys=True, separators=(",", ":")).encode() + b"\n"


def outcome(function, value):
    try:
        return "value", function(value)
    except (ValueError, TypeError) as error:
        return type(error).__name__, str(error)


def histories():
    yield history_at()
    yield history_at_armies()
    yield history_at_armies(player=1)
    yield history_with_necromancy()
    yield market_history()
    yield mandatory_history()
    yield spell_target_history()
    yield map_history(informed_observation(), schema=c.EXPANDED_SCHEMA,
                      episode="synthetic-passive-encoder-exactness")


def reference_detached(value):
    """The pre-fast-path encoder with explicit optional-extension selection.

    Unnegotiated observation extensions are absent from historical wire records;
    a present extension retains its complete typed payload. This reference does
    not delegate serialization or field filtering to the optimized encoder.
    """
    if isinstance(value, tuple):
        return [reference_detached(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        names = c._field_names(type(value))
        if type(value) is c.Observation and value.army_information is None:
            names = tuple(name for name in names if name not in ("army_information", "own_armies"))
        data = {name: reference_detached(getattr(value, name)) for name in names}
        if type(value) is c.Context and value.panel is None and value.kind not in ("exchange", "dialog"):
            del data["panel"], data["selection"]
        elif type(value) is c.Action and value.schema == c.ACTION_SCHEMA:
            for name in ("panel", "selection", "count", "destination"):
                del data[name]
        elif type(value) is c.Window:
            for name in ("town_panel", "armies", "stack_panel", "dialog_panel", "capability_controls", "capability_panel"):
                if getattr(value, name) is None:
                    del data[name]
        elif type(value) is c.Hero and value.in_garrison is None:
            del data["in_garrison"]
        elif type(value) is c.VisibleEvent:
            for name in ("partner", "dialog", "capability"):
                if getattr(value, name) is None:
                    del data[name]
        elif type(value) is c.Offers:
            del data["schema"]
            for name in ("panel", "armies", "stack_panel", "movable_hero_refs", "dialog_panel", "capability_controls", "capability_panel"):
                if getattr(value, name) is None:
                    del data[name]
        elif type(value) is c.TownPanel:
            for name in ("recruitment", "armies", "tavern"):
                if getattr(value, name) is None:
                    del data[name]
        elif type(value) is c.NoticeComponent:
            for name in ("content_id", "skill"):
                if getattr(value, name) is None:
                    del data[name]
        return data
    return value


def shape(value):
    """Exact structure: container and leaf types, key order and identity of every leaf."""
    if type(value) is list:
        return ["list", [shape(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[key, shape(item)] for key, item in value.items()]]
    return [type(value).__qualname__, repr(value), id(value)]


def subtrees(value):
    """Every tuple and dataclass node below value, including value itself."""
    yield value
    if isinstance(value, tuple):
        for item in value:
            yield from subtrees(item)
    elif hasattr(value, "__dataclass_fields__"):
        for name in c._field_names(type(value)):
            yield from subtrees(getattr(value, name))


class Flag(enum.IntEnum):
    ON = 1


class Text(str):
    pass


class Count(int):
    pass


Pair = namedtuple("Pair", "left right")


@dataclass(frozen=True)
class Node:
    label: object
    items: object


class DetachedEncoderTests(unittest.TestCase):
    def test_opted_in_owned_army_subtrees_remain_complete_and_detached(self):
        observed = informed_observation()
        detached = c._wire(observed)
        self.assertEqual(shape(detached), shape(reference_detached(observed)))
        self.assertEqual(detached["army_information"], "own-army-details-v1")
        self.assertEqual(len(detached["own_armies"]), len(observed.own_armies))
        for row, typed in zip(detached["own_armies"], observed.own_armies):
            self.assertEqual(set(row), {field.name for field in fields(c.OwnArmy)})
            self.assertEqual(len(row["slots"]), len(typed.slots))
            for slot, typed_slot in zip(row["slots"], typed.slots):
                self.assertEqual(set(slot), {field.name for field in fields(c.OwnArmyStack)})
                self.assertEqual(slot, {field.name: getattr(typed_slot, field.name)
                                        for field in fields(c.OwnArmyStack)})
        detached["own_armies"][0]["slots"][0]["count"] = 999
        self.assertEqual(observed.own_armies[0].slots[0].count, 14)
        old = replace(observed, army_information=None, own_armies=None)
        self.assertNotIn("army_information", reference_detached(old))
        self.assertNotIn("own_armies", reference_detached(old))

    def test_every_recorded_subtree_encodes_exactly_as_the_reference(self):
        detached, count = c._wire, 0
        for history in histories():
            for frame in history.frames:
                for node in subtrees(frame):
                    self.assertEqual(shape(c._wire(node)), shape(reference_detached(node)))
                    count += 1
                actual = json.dumps(c.encode_frame(frame))  # unsorted: key order is compared too
                c._wire = reference_detached
                try:
                    expected = json.dumps(c.encode_frame(frame))
                finally:
                    c._wire = detached
                self.assertEqual(actual, expected)
        self.assertIs(c._wire, detached)
        self.assertGreater(count, 5000)

    def test_subclassed_and_unusual_leaves_take_the_reference_path(self):
        values = [Flag.ON, Text("a"), Count(3), True, 0, -0.0, float("nan"), float("inf"), "", None, b"raw",
                  Pair(1, Text("b")), (Flag.ON, (Count(2), None)), Node(Text("x"), (Pair(Flag.ON, 1.5), Node(None, ()))),
                  [1, (2,)], {"k": (1,)}, Node, frozenset((1,))]
        for value in values:
            with self.subTest(value=repr(value)):
                self.assertEqual(outcome_any(c._wire, value), outcome_any(reference_detached, value))
        self.assertEqual(outcome_any(c._wire, Node)[0], "TypeError")  # a dataclass type is not a record

    def test_attribute_errors_surface_identically(self):
        @dataclass(frozen=True)
        class Broken:
            first: object
            second: object
        broken = Broken(1, 2)
        object.__delattr__(broken, "second")
        self.assertEqual(outcome_any(c._wire, broken), outcome_any(reference_detached, broken))


def outcome_any(function, value):
    try:
        return "value", shape(function(value))
    except Exception as error:  # the same exception type and message, wherever raised
        return type(error).__name__, str(error)


class WireLineTests(unittest.TestCase):
    def frames(self):
        for history in histories():
            yield from history.frames

    def test_lines_equal_the_reference_encoding_before_and_after_caching(self):
        count = 0
        for frame in self.frames():
            c.release_canonical((frame,))
            expected = reference_wire(frame)
            self.assertEqual(replay_history._wire(frame), expected)  # cold: canonical text is computed here
            self.assertEqual(replay_history._wire(frame), expected)  # warm: served by the frame's cache
            self.assertEqual(c.canonical_frame(frame).encode() + b"\n", expected)
            count += 1
        self.assertGreater(count, 50)

    def test_rejected_inputs_fail_or_encode_exactly_as_the_reference(self):
        frame = next(self.frames())
        c.release_canonical((frame,))
        nan = replace(frame)  # a fresh copy without cached text
        object.__setattr__(nan, "day", float("nan"))  # bypasses validation: only the encoder decides
        for value in (nan, None, {"kind": "decision"}, history_at()):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(outcome(replay_history._wire, value), outcome(reference_wire, value))
        self.assertIn(b"NaN", replay_history._wire(nan))
        self.assertNotIn("_canonical_json", nan.__dict__)  # never cached a text canonical_frame refuses


def reference_unique(message):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(message.format(key))
            result[key] = value
        return result
    return unique


class StrictJsonTests(unittest.TestCase):
    TEXTS = ['{"a":1,"b":{"c":[1,2,{"d":null}],"e":true}}', '{"b":1,"a":2}', '{}', '[{"x":1},{"x":2}]',
             '{"a":1,"a":2}', '{"a":{"b":1,"c":2,"b":3},"d":1}', '{"a":1,"b":2,"a":1,"b":2}', '{"a":NaN}',
             '{"a":1.5,"b":-0.0,"c":1e999}', '{"\u00e9":1,"\u0065\u0301":2}', '[1,2', '"text"']

    def test_parsers_return_or_raise_exactly_as_the_reference(self):
        for parse, hook, message in ((contextual_evidence.strict_json, "_unique", "Duplicate evidence field: {}"),
                                     (contextual_protocol.loads, "unique_object", "Duplicate JSON field: {}")):
            module = contextual_evidence if hook == "_unique" else contextual_protocol
            original = getattr(module, hook)
            for text in self.TEXTS:
                with self.subTest(text=text, hook=hook):
                    actual = outcome(parse, text)
                    setattr(module, hook, reference_unique(message))
                    try:
                        expected = outcome(parse, text)
                    finally:
                        setattr(module, hook, original)
                    self.assertEqual(actual, expected)
                    if actual[0] == "value":
                        self.assertEqual(json.dumps(actual[1]), json.dumps(expected[1]))  # key order too


if __name__ == "__main__":
    unittest.main()
