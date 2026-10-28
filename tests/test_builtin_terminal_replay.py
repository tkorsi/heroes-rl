"""Synthetic request accounting; these tests do not execute native games."""
from copy import deepcopy
from itertools import product
import unittest

from experiments.headless import builtin_replay as replay
from tests import test_builtin_replay as fixtures


class TerminalAttemptTests(fixtures.WireTests):
    def tail_fixture(self, player=0, pack_type=182):
        self.fixture(player)
        self.server[1]["client_out"] = 0
        self.server[3]["client_out"] = 5
        self.server[5].update(client_out=10, boundary="game_end", terminals=[
            {"player": 0, "victory": False}, {"player": 1, "victory": True}])
        self.tape[-1]["client_packets"] = 10
        self.tape[-1:-1] = [
            {"kind": "enter", "actor": "client", "operation": "request", "packet": {
                "index": 7, "completed": 7, "phase": "callbacks_complete", "type": 117}},
            {"kind": "request", "player": 1-player, "request_id": 5, "pack_type": pack_type,
             "pack": "deadbeef", "source": "native_opponent"},
            {"kind": "leave", "actor": "client", "operation": "request"}]
        self.renumber()

    def test_tail_is_preserved_and_separately_counted_for_both_learner_seats(self):
        for player, pack_type in product((0, 1), (180, 182, 185, 190, 193)):
            self.tail_fixture(player, pack_type)
            original = deepcopy(self.tape)
            wire = self.check()
            self.assertEqual(self.tape, original)
            self.assertEqual(wire["requests"], 3)
            self.assertEqual(wire["native_opponent_requests"], 2)
            accounting = wire["request_accounting"]
            self.assertEqual(accounting["authoritative_inputs"], 2)
            self.assertEqual(accounting["terminal_unapplied"][0]["pack"], "deadbeef")
            self.assertEqual(accounting["terminal_unapplied"][0]["pack_type"], pack_type)
            self.assertEqual(accounting["final_terminal_input"], {"input_seq": 1, "player": 1-player,
                "request_id": 4, "pack_type": 91, "first_client_packet": 6, "last_client_packet": 10})

    def test_tail_never_excuses_an_omitted_or_changed_authoritative_request(self):
        for player, pack_type in product((0, 1), (180, 182, 185, 190, 193)):
            for change in (lambda t: t.pop(7), lambda t: t[7].update(pack="ccee"),
                           lambda t: t[7].update(request_id=5), lambda t: t[4].update(player=1-player)):
                self.tail_fixture(player, pack_type); change(self.tape); self.renumber()
                with self.subTest(player=player, pack_type=pack_type, change=change), self.assertRaises(ValueError):
                    self.check()

    def test_tail_strict_identity_and_final_packet_interval(self):
        changes = [lambda t: t[-3].update(source="automatic"), lambda t: t[-3].update(source="adapter"),
                   lambda t: t[-3].update(player=self.manifest["learner_player"]),
                   lambda t: t[-3].update(pack_type=198),
                   lambda t: t[-3].update(pack_type=183),
                   lambda t: t[-3].update(pack_type=True),
                   lambda t: t[-3].update(request_id=6), lambda t: t[-4].update(operation="packet"),
                   lambda t: t[-4]["packet"].update(index=5, completed=5),
                   lambda t: t[-4]["packet"].update(index=10, completed=10),
                   lambda t: t[-4]["packet"].update(index=True, completed=True),
                   lambda t: t[-4]["packet"].update(completed=6),
                   lambda t: t[-4]["packet"].update(phase="applying"),
                   lambda t: t[-4]["packet"].update(type=False),
                   lambda t: t[-1].update(client_packets=9)]
        for player, pack_type in product((0, 1), (180, 182, 185, 190, 193)):
            for change in changes:
                self.tail_fixture(player, pack_type); change(self.tape)
                with self.subTest(player=player, pack_type=pack_type, change=change), self.assertRaises(ValueError):
                    self.check()

    def test_only_one_attempt_after_actual_complementary_natural_end(self):
        for player, pack_type in product((0, 1), (180, 182, 185, 190, 193)):
            for change in (lambda: self.server[-1].update(boundary="battle_end"),
                           lambda: self.server[-1].update(terminals=[]),
                           lambda: self.server[-1]["terminals"][0].update(victory=True),
                           lambda: self.server[-1]["terminals"][0].update(victory=0),
                           lambda: self.server[-1]["terminals"][0].update(player=False),
                           lambda: self.manifest.update(status="truncated"),
                           lambda: self.tape.insert(-2, deepcopy(self.tape[-3])),
                           lambda: self.tape.pop()):
                self.tail_fixture(player, pack_type); change(); self.renumber()
                with self.subTest(player=player, pack_type=pack_type, change=change), self.assertRaises(ValueError):
                    self.check()

    def test_later_learner_choice_is_not_a_terminal_tail(self):
        for player, pack_type in product((0, 1), (180, 182, 185, 190, 193)):
            self.tail_fixture(player, pack_type)
            self.tape[-1:-1] = [{"kind":"enter", "actor":f"player{player}", "operation":"worker", "packet":None},
                               {"kind":"response", "player":player, "response":{}},
                               {"kind":"leave", "actor":f"player{player}", "operation":"worker"}]
            self.renumber()
            with self.subTest(player=player, pack_type=pack_type), self.assertRaisesRegex(ValueError, "Learner response follows"):
                self.check()

    def test_unreviewed_adjacent_packet_types_still_fail_closed(self):
        supported = {180, 182, 185, 190, 193}
        for player, pack_type in product((0, 1), set(range(180, 200)) - supported):
            self.tail_fixture(player, pack_type)
            with self.subTest(player=player, pack_type=pack_type), self.assertRaisesRegex(
                    ValueError, "Only terminal native-opponent"):
                self.check()


class TerminalReportTests(fixtures.NativeReportTests):
    def add_accounting(self, pack_type=182):
        self.wire["requests"] = 5
        for key in ("loaded", "checked"):
            self.native["summary"][key]["requests"] = 5
        accounting = {"authoritative_inputs":4, "terminal_unapplied":[{
            "seq":20, "player":0, "request_id":43, "pack_type":pack_type, "pack":"deadbeef", "source":"native_opponent",
            "receipt":{"index":8, "completed":8, "phase":"callbacks_complete", "type":117}}],
            "final_terminal_input":{"input_seq":3, "player":0, "request_id":42, "pack_type":90,
                                    "first_client_packet":7, "last_client_packet":10}}
        self.wire["request_accounting"] = deepcopy(accounting)
        self.native["summary"]["request_accounting"] = deepcopy(accounting)

    def test_explicit_accounting_does_not_claim_tail_as_server_input(self):
        for pack_type in (180, 182, 185, 190, 193):
            self.setUp(); self.add_accounting(pack_type); self.check()
            self.assertEqual(self.native["summary"]["server"]["inputs"], 4)

    def test_nonzero_tail_needs_exact_native_accounting_and_origin(self):
        for pack_type in (180, 182, 185, 190, 193):
            for change in (lambda n:n["summary"].pop("request_accounting"),
                           lambda n:n["summary"]["request_accounting"]["terminal_unapplied"].clear(),
                           lambda n:n["summary"]["request_accounting"]["terminal_unapplied"][0].update(pack="aabb"),
                           lambda n:n["summary"]["request_accounting"]["terminal_unapplied"][0].update(pack_type=198),
                           lambda n:n["summary"]["request_accounting"].update(authoritative_inputs=5),
                           lambda n:n["summary"]["server"].update(inputs=5)):
                self.setUp(); self.add_accounting(pack_type); change(self.native)
                with self.subTest(pack_type=pack_type, change=change), self.assertRaises(ValueError):
                    self.check()
            self.setUp(); self.add_accounting(pack_type)
            for accounting in (self.wire["request_accounting"], self.native["summary"]["request_accounting"]):
                accounting["final_terminal_input"]["request_id"] = 41
            with self.assertRaisesRegex(ValueError, "barrier origin"): self.check()

