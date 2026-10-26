"""Per-episode verification shortcuts must produce exactly the reference bytes and errors."""
from dataclasses import replace
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
