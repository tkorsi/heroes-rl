"""Synthetic request accounting; these tests do not execute native games."""
from copy import deepcopy
import unittest

from experiments.headless import builtin_replay as replay
from tests import test_builtin_replay as fixtures


class TerminalAttemptTests(fixtures.WireTests):
    def tail_fixture(self, player=0):
        self.fixture(player)
        self.server[1]["client_out"] = 0
        self.server[3]["client_out"] = 5
        self.server[5].update(client_out=10, boundary="game_end", terminals=[
            {"player": 0, "victory": False}, {"player": 1, "victory": True}])
        self.tape[-1]["client_packets"] = 10
        self.tape[-1:-1] = [
            {"kind": "enter", "actor": "client", "operation": "request", "packet": {
                "index": 7, "completed": 7, "phase": "callbacks_complete", "type": 117}},
            {"kind": "request", "player": 1-player, "request_id": 5, "pack_type": 182,
             "pack": "deadbeef", "source": "native_opponent"},
            {"kind": "leave", "actor": "client", "operation": "request"}]
        self.renumber()

    def test_tail_is_preserved_and_separately_counted_for_both_learner_seats(self):
        for player in (0, 1):
            self.tail_fixture(player)
            original = deepcopy(self.tape)
            wire = self.check()
            self.assertEqual(self.tape, original)
            self.assertEqual(wire["requests"], 3)
            self.assertEqual(wire["native_opponent_requests"], 2)
            accounting = wire["request_accounting"]
            self.assertEqual(accounting["authoritative_inputs"], 2)
            self.assertEqual(accounting["terminal_unapplied"][0]["pack"], "deadbeef")
            self.assertEqual(accounting["final_terminal_input"], {"input_seq": 1, "player": 1-player,
                "request_id": 4, "pack_type": 91, "first_client_packet": 6, "last_client_packet": 10})

    def test_tail_never_excuses_an_omitted_or_changed_authoritative_request(self):
        for change in (lambda t: t.pop(7), lambda t: t[7].update(pack="ccee"),
                       lambda t: t[7].update(request_id=5), lambda t: t[4].update(player=1)):
            self.tail_fixture(); change(self.tape); self.renumber()
            with self.assertRaises(ValueError): self.check()

    def test_tail_strict_identity_and_final_packet_interval(self):
        changes = [lambda t: t[-3].update(source="automatic"), lambda t: t[-3].update(source="adapter"),
                   lambda t: t[-3].update(player=0), lambda t: t[-3].update(pack_type=198),
                   lambda t: t[-3].update(request_id=6), lambda t: t[-4].update(operation="packet"),
                   lambda t: t[-4]["packet"].update(index=5, completed=5),
                   lambda t: t[-4]["packet"].update(index=10, completed=10),
                   lambda t: t[-4]["packet"].update(index=True, completed=True),
                   lambda t: t[-4]["packet"].update(completed=6),
                   lambda t: t[-4]["packet"].update(phase="applying"),
                   lambda t: t[-4]["packet"].update(type=False),
                   lambda t: t[-1].update(client_packets=9)]
        for change in changes:
            self.tail_fixture(); change(self.tape)
            with self.subTest(change=change), self.assertRaises(ValueError): self.check()

    def test_only_one_attempt_after_actual_complementary_natural_end(self):
        for change in (lambda: self.server[-1].update(boundary="battle_end"),
                       lambda: self.server[-1].update(terminals=[]),
                       lambda: self.server[-1]["terminals"][0].update(victory=True),
                       lambda: self.server[-1]["terminals"][0].update(victory=0),
                       lambda: self.server[-1]["terminals"][0].update(player=False),
                       lambda: self.manifest.update(status="truncated"),
                       lambda: self.tape.insert(-2, deepcopy(self.tape[-3]))):
            self.tail_fixture(); change(); self.renumber()
            with self.assertRaises(ValueError): self.check()

    def test_later_learner_choice_is_not_a_terminal_tail(self):
        self.tail_fixture()
        self.tape[-1:-1] = [{"kind":"enter", "actor":"player0", "operation":"worker", "packet":None},
                           {"kind":"response", "player":0, "response":{}},
                           {"kind":"leave", "actor":"player0", "operation":"worker"}]
        self.renumber()
        with self.assertRaisesRegex(ValueError, "Learner response follows"): self.check()


class TerminalReportTests(fixtures.NativeReportTests):
    def add_accounting(self):
        self.wire["requests"] = 5
        for key in ("loaded", "checked"):
            self.native["summary"][key]["requests"] = 5
        accounting = {"authoritative_inputs":4, "terminal_unapplied":[{
            "seq":20, "player":0, "request_id":43, "pack_type":182, "pack":"deadbeef", "source":"native_opponent",
            "receipt":{"index":8, "completed":8, "phase":"callbacks_complete", "type":117}}],
            "final_terminal_input":{"input_seq":3, "player":0, "request_id":42, "pack_type":90,
                                    "first_client_packet":7, "last_client_packet":10}}
        self.wire["request_accounting"] = deepcopy(accounting)
        self.native["summary"]["request_accounting"] = deepcopy(accounting)

    def test_explicit_accounting_does_not_claim_tail_as_server_input(self):
        self.add_accounting(); self.check()
        self.assertEqual(self.native["summary"]["server"]["inputs"], 4)

    def test_nonzero_tail_needs_exact_native_accounting_and_origin(self):
        for change in (lambda n:n["summary"].pop("request_accounting"),
                       lambda n:n["summary"]["request_accounting"]["terminal_unapplied"].clear(),
                       lambda n:n["summary"]["request_accounting"]["terminal_unapplied"][0].update(pack="aabb"),
                       lambda n:n["summary"]["request_accounting"].update(authoritative_inputs=5),
                       lambda n:n["summary"]["server"].update(inputs=5)):
            self.setUp(); self.add_accounting(); change(self.native)
            with self.assertRaises(ValueError): self.check()
        self.setUp(); self.add_accounting()
        for accounting in (self.wire["request_accounting"], self.native["summary"]["request_accounting"]):
            accounting["final_terminal_input"]["request_id"] = 41
        with self.assertRaisesRegex(ValueError, "barrier origin"): self.check()

