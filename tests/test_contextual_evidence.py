"""Synthetic barrier corruption checks; these are not real-game evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.headless import contextual_evidence as evidence


def write_barriers(root, *, outcomes=None):
    server = {"schema": evidence.SCHEMA, "kind": "server", "episode": "1",
              "interface": "adventure-windows-v1", "barrier_id": 1, "origin": "request",
              "request": {"player": 1, "id": 42, "type": 90}, "client_packets": 123,
              "outcomes": outcomes or ["win", "loss"]}
    client = {**deepcopy(server), "kind": "client", "callbacks_complete": True, "observed_count": 123}
    (Path(root) / "engine").mkdir(exist_ok=True)
    for name, record in zip(evidence.FILES, (server, client)):
        (Path(root) / name).write_text(json.dumps(record) + "\n")
    return server, client


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_barriers(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, outcomes=None):
        return evidence.terminal_barrier(self.root, episode="1", interface="adventure-windows-v1",
                                         outcomes=outcomes or ["win", "loss"])

    def test_matching_both_colors_bind_exact_file_bytes(self):
        for outcomes in (["win", "loss"], ["loss", "win"]):
            write_barriers(self.root, outcomes=outcomes)
            result = self.validate(outcomes)
            self.assertTrue(result["verified"])
            self.assertEqual(result["files"], {name: evidence.file_identity(self.root / name) for name in evidence.FILES})

    def test_armies_profile_barrier_requires_both_exact_versioned_receipts(self):
        server, client = write_barriers(self.root)
        for name, record in zip(evidence.FILES, (server, client)):
            record["interface"] = "adventure-armies-v1"
            (self.root / name).write_text(json.dumps(record) + "\n")
        result = evidence.terminal_barrier(self.root, episode="1", interface="adventure-armies-v1",
                                           outcomes=["win", "loss"])
        self.assertTrue(result["verified"])
        with self.assertRaises(ValueError): self.validate()

    def test_missing_duplicate_partial_and_unknown_fields_fail(self):
        for name in evidence.FILES:
            for mode in ("missing", "duplicate", "partial", "unknown", "duplicate_key"):
                with self.subTest(name=name, mode=mode):
                    write_barriers(self.root)
                    path = self.root / name
                    raw = path.read_text()
                    if mode == "missing": path.unlink()
                    elif mode == "duplicate": path.write_text(raw + raw)
                    elif mode == "partial": path.write_text(raw.rstrip())
                    elif mode == "unknown": path.write_text(raw.replace('{', '{"secret":1,', 1))
                    else: path.write_text(raw.replace('{', '{"schema":"wrong",', 1))
                    with self.assertRaises((ValueError, OSError)): self.validate()

    def test_identity_request_and_callback_mismatches_fail(self):
        mutations = [lambda r: r.update(episode="2"), lambda r: r.update(interface="adventure-town-v1"),
                     lambda r: r.update(barrier_id=True), lambda r: r.update(origin="start"),
                     lambda r: r.update(origin="tick"), lambda r: r.update(outcomes=["loss", "win"]),
                     lambda r: r.update(client_packets=124), lambda r: r.update(client_packets=True),
                     lambda r: r["request"].update(id=43), lambda r: r["request"].update(player=True),
                     lambda r: r["request"].update(type=0), lambda r: r.update(callbacks_complete=False),
                     lambda r: r.update(observed_count=122)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                _, client = write_barriers(self.root)
                mutate(client)
                (self.root / evidence.FILES[1]).write_text(json.dumps(client) + "\n")
                with self.assertRaises(ValueError): self.validate()

    def test_rehashed_matching_but_invalid_request_is_rejected(self):
        server, client = write_barriers(self.root)
        for record in (server, client): record["request"]["id"] = -1
        for name, record in zip(evidence.FILES, (server, client)):
            (self.root / name).write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError): self.validate()

    def test_neutral_origin_requires_the_native_battle_action_type(self):
        for player, pack_type, accepted in ((-1, 198, True), (-1, 90, False), (-2, 198, False), (2, 198, False)):
            with self.subTest(player=player, pack_type=pack_type):
                server, client = write_barriers(self.root)
                for name, record in zip(evidence.FILES, (server, client)):
                    record["request"].update(player=player, type=pack_type)
                    (self.root / name).write_text(json.dumps(record) + "\n")
                if accepted:
                    self.assertTrue(self.validate()["verified"])
                else:
                    with self.assertRaises(ValueError): self.validate()

    def test_path_escape_and_symlinks_are_rejected(self):
        for name in ("../manifest.json", "/tmp/file"):
            with self.assertRaises(ValueError): evidence.inside(self.root, name)
        path = self.root / evidence.FILES[1]
        path.unlink()
        path.symlink_to(self.root / evidence.FILES[0])
        with self.assertRaises(ValueError): self.validate()

    def test_natural_close_must_be_final_complete_and_match_receipt(self):
        barrier = self.validate()
        path = self.root / "engine/contextual-replay.jsonl"
        row = {"seq": 12, "kind": "close", "barrier_id": 1, "client_packets": 123}
        raw = json.dumps(row) + "\n"
        path.write_text('{"kind":"leave"}\n' + raw)
        self.assertEqual(evidence.terminal_capture_close(self.root, barrier), row)
        for bad in (raw.rstrip(), raw + '{"kind":"enter"}\n', raw.replace('123', '124'),
                    raw.replace('"barrier_id": 1', '"barrier_id": true')):
            with self.subTest(bad=bad):
                path.write_text(bad)
                with self.assertRaises(ValueError): evidence.terminal_capture_close(self.root, barrier)


if __name__ == "__main__":
    unittest.main()
