"""Cached canonical frame text must reproduce every original encoding exactly."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner import contextual as c
from tests.test_contextual_dialogs import FAMILIES, history_at, panel
from tests.test_contextual_rollout import finish_pair, initial, resolve


def reference_encode(frame):
    """The original encoder: wire every field, then the observation again."""
    data = c._wire(frame)
    if frame.observation is not None:
        data["observation"] = c.encode_observation(frame.observation, schema=frame.schema)
    return data


def reference_text(frame):
    return json.dumps(reference_encode(frame), sort_keys=True, separators=(",", ":"))


def reference_digest(frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(reference_text(frame).encode() + b"\n")
    return digest.hexdigest()


def finalized_pairs():
    pairs = []
    for boundary in ("natural", "cutoff", "failure"):
        pair = tuple(initial(player, schema=c.DIALOGS_SCHEMA) for player in (0, 1))
        for history in pair:
            resolve(history, c.Action("end_turn", schema=c.DIALOGS_ACTION_SCHEMA))
        finish_pair(pair, boundary)
        pairs.append(pair)
    return pairs


class CanonicalFrameCacheTests(unittest.TestCase):
    def histories(self):
        result = [history_at(panel(family)) for family in FAMILIES]
        return result + [history for pair in finalized_pairs() for history in pair]

    def test_cached_text_and_digest_match_the_original_encoder(self):
        for history in self.histories():
            frames = history.frames
            for frame in frames:
                self.assertEqual(c.encode_frame(frame), reference_encode(frame))
                expected = reference_text(frame)
                self.assertEqual(c.canonical_frame(frame), expected)
                self.assertEqual(c.canonical_frame(frame), expected)  # cached path
            self.assertEqual(history.digest(), reference_digest(frames))

    def test_cache_is_invisible_to_equality_replace_and_decoding(self):
        frame = history_at(panel()).frames[-1]
        text = c.canonical_frame(frame)
        decoded = c.decode_frame(json.loads(text))
        self.assertNotIn("_canonical_json", decoded.__dict__)
        self.assertEqual(decoded, frame)
        self.assertEqual(c.canonical_frame(decoded), text)
        later = replace(frame, day=frame.day + 1)
        self.assertNotIn("_canonical_json", later.__dict__)
        self.assertEqual(c.canonical_frame(later), reference_text(later))
        self.assertNotEqual(c.canonical_frame(later), text)
        with self.assertRaises(ValueError):
            c.canonical_frame(json.loads(text))

    def test_rollout_history_serialization_is_unchanged_and_detached(self):
        from experiments.learner.contextual_rollout import _frame_bytes
        for history in self.histories():
            for frame in history.frames:
                self.assertEqual(_frame_bytes(frame), reference_text(frame).encode() + b"\n")

    def test_auditor_digests_match_the_contract_reference(self):
        from validation.probes.audit_contextual_histories import audit_player
        with tempfile.TemporaryDirectory() as folder:
            for index, pair in enumerate(finalized_pairs()):
                for history in pair:
                    path = Path(folder) / f"history-{index}-{history.player}.jsonl"
                    path.write_text("".join(reference_text(frame) + "\n" for frame in history.frames))
                    report, rebuilt = audit_player(path, history.player)
                    self.assertFalse(report["failures"])
                    frames = history.frames
                    prefix = frames[:-1] if frames[-1].kind in ("finalized", "cutoff_finalized") else frames
                    self.assertEqual(report["canonical_sha256"], reference_digest(frames))
                    self.assertEqual(report["prefix_sha256"], reference_digest(prefix))
                    self.assertEqual(rebuilt.frames, frames)


class VerbatimRedeliveryTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch": raise
            self.skipTest("Optional torch is not installed")
        from experiments.learner import contextual_model as m
        from experiments.learner.contextual_rollout import ContextualSession
        torch.manual_seed(11)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.history = history_at(panel())
        self.records = [json.loads(reference_text(frame)) for frame in self.history.frames]
        self.session = ContextualSession(model, self.history.episode, 0, "fixed", "opponent", schema=c.DIALOGS_SCHEMA)
        self.session.consume(json.loads(json.dumps(self.records)))

    def original_outcome(self, raw):
        try:
            frame = c.decode_frame(raw)
        except ValueError:
            return "rejected"
        return "kept" if self.session.history.frames[frame.seq] == frame else "rejected"

    def outcome(self, raw):
        session = self.session
        before = (session.history.frames, session._broken)
        session._broken = False
        try:
            session.consume([raw])
            result = "kept"
        except ValueError:
            result = "rejected"
        self.assertEqual(session.history.frames, before[0])
        session._broken = before[1]
        return result

    def test_redelivered_prefix_skips_decoding_without_changing_history(self):
        from unittest import mock
        self.assertEqual(self.session._verbatim, set(range(len(self.records))))
        with mock.patch.object(c, "decode_frame", side_effect=AssertionError("decoded")):
            self.session.consume(json.loads(json.dumps(self.records)))
        self.assertEqual(self.session.history.frames, self.history.frames)

    def test_every_other_record_takes_the_original_path(self):
        decision = self.records[-1]
        variants = []
        changed = json.loads(json.dumps(decision)); changed["day"] += 1; variants.append(changed)
        boolean = json.loads(json.dumps(self.records[0])); boolean["player"] = False; variants.append(boolean)
        tupled = json.loads(json.dumps(decision)); tupled["observation"]["tiles"] = tuple(tupled["observation"]["tiles"])
        variants.append(tupled)
        nested = json.loads(json.dumps(decision)); nested["observation"]["size"] = tuple(nested["observation"]["size"])
        variants.append(nested)
        extra = json.loads(json.dumps(decision)); extra["unknown"] = None; variants.append(extra)
        floated = json.loads(json.dumps(decision)); floated["day"] = float(floated["day"]); variants.append(floated)
        for raw in variants:
            with self.subTest(raw=sorted(raw)):
                self.assertEqual(self.outcome(raw), self.original_outcome(raw))

    def test_unproven_positions_are_decoded(self):
        self.session._verbatim.clear()
        from unittest import mock
        with mock.patch.object(c, "decode_frame", wraps=c.decode_frame) as decoded:
            self.session.consume(json.loads(json.dumps(self.records)))
        self.assertEqual(decoded.call_count, len(self.records))


if __name__ == "__main__":
    unittest.main()
