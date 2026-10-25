"""In-process engine module contracts; no Heroes Data, game rules or gameplay.

Conversion tests compare the module with the socket backend's own codec:
JsonWriter text with newlines removed, decoded by the adapter's json.loads,
and json.dumps responses parsed by VCMI's strict JSON parser.
"""
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from experiments.headless.contextual_protocol import loads
from experiments.headless import embedded_evidence as ev
from tests.test_native_session import native_include_flags

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "vcmi/build"


def load_module():
    try:
        path = ev.module_path(BUILD / "bin")
    except ValueError:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(ev.MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def same(left, right):
    """Equal values with equal JSON types (bool, int and float are distinct)."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return left == right and math.copysign(1, left) == math.copysign(1, right)
    return left == right


def socket_decode(text):
    line = text.replace(b"\n", b"").replace(b"\r", b"")
    try:
        return loads(line), None
    except ValueError as error:  # JSONDecodeError and UnicodeDecodeError included
        return None, type(error).__name__


FLOATS = [0.1, 1 / 3, 2.5, -1.5, 100.0, 123456.0, 1e6, 123456789.0, 1e16, 2.0 ** 53, 1e21, 1e-5, 1e-7,
          5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, -0.0, 0.0, 7.0, 0.30000000000000004]
FLOATS += [struct.unpack("<d", struct.pack("<Q", (index * 0x9E3779B97F4A7C15) & 0x7FEFFFFFFFFFFFFF))[0]
           for index in range(1, 400)]
STRINGS = [b"", b"plain", b"a\"b", b"back\\slash", b"trail\\", b"esc\\n", b"\\u0041", b"\\\\\"", b"tab\tx",
           b"nl\nx", b"cr\rx", b"ctl\x01x", b"nul\x00x", b"del\x7f", b"\xc3\xa9", b"\xed\xa0\x80", b"\xff\xfe",
           b"\xf0\x9f\x8f\xb0 castle", b"\\b\\f\\r\\t", b"\b\f", b"quote\\\"end"]


@unittest.skipIf(MODULE is None, "requires heroes_embedded_engine built for this Python (-DENABLE_EMBEDDED_ENGINE=ON)")
class StreamConversionTests(unittest.TestCase):
    def test_floats_match_jsonwriter_then_json_loads(self):
        for value in FLOATS:
            with self.subTest(value=value):
                converted, text, refusal = MODULE.stream_value("float", value)
                expected, error = socket_decode(text)
                if error:
                    self.assertIsNotNone(refusal)
                    self.assertIsNone(converted)
                else:
                    self.assertIsNone(refusal)
                    self.assertTrue(same(converted, expected), (converted, expected, text))
        for value in (float("inf"), float("-inf"), float("nan")):
            converted, text, refusal = MODULE.stream_value("float", value)
            self.assertEqual(socket_decode(text)[0], None)
            self.assertIn("ValueError", refusal)

    def test_strings_match_jsonwriter_escaping_then_json_loads(self):
        for value in STRINGS:
            with self.subTest(value=value):
                converted, text, refusal = MODULE.stream_value("string", value)
                expected, error = socket_decode(text)
                self.assertEqual(refusal is not None, error is not None, (value, refusal, error))
                if error is None:
                    self.assertTrue(same(converted, expected), (converted, expected))

    def test_responses_match_json_dumps_then_strict_vcmi_parser(self):
        corpus = [{"protocol_version": 9, "type": "context_action", "episode": "12", "player": 1,
                   "decision_id": 41, "revision": 3, "action": {"kind": "move", "x": 12, "y": 7}},
                  {"a": [1, 2.5, True, False, None, "x", {"z": -1, "é": "\"\\\n"}], "b": {}, "c": []},
                  {"float": 1.0, "small": 1e-7, "big": 1e300, "neg": -0.5, "int": -(2 ** 63), "max": 2 ** 63 - 1},
                  {"unicode": "\U0001F3F0 castle", "tabs": "\t\b\f\r"}]
        refused = []
        for value in corpus:
            with self.subTest(value=value):
                line = json.dumps(value, separators=(",", ":"))
                try:
                    parsed = MODULE.parsed_text(line)
                except MODULE.EngineError:
                    refused.append(value)  # the socket backend refuses it too; submit_text uses this parser
                    continue
                self.assertEqual(MODULE.json_text(value), parsed)
        # json.dumps escapes non-ASCII as \uXXXX, which VCMI's strict parser refuses.
        self.assertEqual(refused, corpus[1:2] + corpus[3:4])
        for bad, kind in (({"x": float("nan")}, ValueError), ({"x": 2 ** 64}, OverflowError), ({1: 2}, TypeError),
                          ({"x": object()}, TypeError)):
            with self.subTest(bad=bad), self.assertRaises(kind):
                MODULE.json_text(bad)
        deep = current = {}
        for _ in range(70):
            current["d"] = {}
            current = current["d"]
        with self.assertRaises(ValueError):
            MODULE.json_text(deep)

    def test_lifecycle_refusals_without_assets(self):
        script = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location({ev.MODULE_NAME!r}, {str(ev.module_path(BUILD / 'bin'))!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "e = m.Engine()\n"
            "checks = []\n"
            "for call in (lambda: e.take(0.0), lambda: e.submit(1, 1, {}), lambda: e.submit_text(1, 1, '{}'),\n"
            "             lambda: e.finish(natural=False, timeout=0.0)):\n"
            "    try: call(); checks.append('accepted')\n"
            "    except m.EngineError: checks.append('refused')\n"
            "try: e.reset(episode='1', map='Maps/x.vmap', capture_directory='/nonexistent-new', seed=1, learner=0)\n"
            "except (m.EngineError, ValueError): checks.append('refused')\n"
            "try: e.finish(timeout=1.0)\n"
            "except TypeError: checks.append('refused')\n"
            "try: m.Engine(); checks.append('second')\n"
            "except m.EngineError: checks.append('refused')\n"
            "print(e.poisoned, e.active, checks)\n"
            "e.close()\n"
            "try: e.take(0.0)\n"
            "except m.EngineError as error: print('closed:', error)\n")
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-I", "-c", script], cwd=temporary, capture_output=True,
                                    text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("False False ['refused', 'refused', 'refused', 'refused', 'refused', 'refused', 'refused']", result.stdout)
        self.assertIn("closed: Engine is closed", result.stdout)


@unittest.skipUnless(shutil.which("g++") or shutil.which("clang++"), "requires a C++20 compiler")
class IncrementalChannelTests(unittest.TestCase):
    def test_take_submit_drain_terminal_failure_and_high_water(self):
        engine = ROOT / "vcmi"
        library = engine / "build/bin"
        if not any(library.glob("libvcmi.*")):
            self.skipTest("requires prepared and rebuilt VCMI shared library")
        compiler = shutil.which("g++") or shutil.which("clang++")
        with tempfile.TemporaryDirectory(prefix="native-take-") as temporary:
            binary = Path(temporary) / "take-contract"
            command = [compiler, "-std=gnu++20", "-pthread", *native_include_flags(engine),
                       str(ROOT / "tests/native/session_take.cpp"), "-L" + str(library),
                       "-lvcmi", "-Wl,-rpath," + str(library), "-o", str(binary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("incremental native channel contract passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
