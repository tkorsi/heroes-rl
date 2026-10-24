"""Strict diagnostic evidence checks, without launching or simulating an engine."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile

from experiments.learner.contextual import Frame, Notice, NoticeComponent, NoticeToken, Result, VisibleEvent
from validation.probes import create_casualty_fixture as fixtures
from validation.probes import validate_contextual_encounters as encounters
from validation.probes import validate_town_native as town


def frame(seq, kind, **fields):
    return Frame("diagnostic-evidence", 0, seq, 1, kind, **fields)


def resource_evidence():
    target = SimpleNamespace(handle="resource", kind=79, position=(4, 10, 0))
    hero = SimpleNamespace(handle="own")
    before = SimpleNamespace(resources=(0, 0, 0, 0, 0, 0, 100), objects=(target,), heroes=(hero,))
    after = SimpleNamespace(resources=(0, 0, 0, 0, 0, 0, 600), objects=(), heroes=(hero,))
    result = Result("interaction", True, (4, 10, 0), 800, False)
    notice = Notice("info", (NoticeToken("append_text", text_id="core.advevent.113"),
                              NoticeToken("replace_text", text_id="core.restypes.6")),
                    (NoticeComponent("resource", resource="gold", amount=500),))
    frames = [frame(1, "consumed", decision_seq=0, consumer="engine"),
              frame(2, "issued", decision_seq=0, command_index=0),
              frame(3, "event", event=VisibleEvent("visit", subject=target.handle, object_kind=79)),
              frame(4, "event", event=VisibleEvent("notice", notice=notice)),
              frame(5, "acknowledged", decision_seq=0, command_index=0, accepted=True),
              frame(6, "result", decision_seq=0, result=result)]
    return target, before, after, frames, result


class ContextualGapProbeTests(unittest.TestCase):
    def test_actual_resource_limit_and_positive_stock_zero_limit(self):
        cost = (0, 0, 0, 0, 0, 0, 60)
        evidence = town.recruitment_budget_evidence(14, cost, (0, 0, 0, 0, 0, 0, 475), 7)
        self.assertEqual(evidence["invalid_count"], 8)
        self.assertLessEqual(evidence["invalid_count"], evidence["stock"])
        exhausted = town.recruitment_budget_evidence(7, cost, (0, 0, 0, 0, 0, 0, 55), 0)
        self.assertEqual(exhausted["invalid_count"], 1)

    def test_empty_stock_stock_limit_and_wrong_slider_do_not_prove_affordability(self):
        for stock, gold, slider in ((0, 0, 0), (14, 840, 14), (14, 475, 6), (14, 475, 8)):
            with self.subTest(stock=stock, gold=gold, slider=slider), self.assertRaises(encounters.base.ProbeFailure):
                town.recruitment_budget_evidence(stock, (0, 0, 0, 0, 0, 0, 60),
                                                 (0, 0, 0, 0, 0, 0, gold), slider)

    def test_selected_multiresource_price_is_used(self):
        evidence = town.recruitment_budget_evidence(10, (2, 0, 0, 0, 0, 0, 100),
                                                    (5, 0, 0, 0, 0, 0, 900), 2)
        self.assertEqual(evidence["invalid_count"], 3)

    def test_required_native_conditions_cannot_be_missing_unverified_or_contradictory(self):
        encounters.require_checks([{"check": "casualties", "status": "passed"}], ["casualties"])
        for rows in ([], [{"check": "casualties", "status": "unverified"}],
                     [{"check": "casualties", "status": "failed"}],
                     [{"check": "casualties", "status": "passed"},
                      {"check": "casualties", "status": "failed"}]):
            with self.subTest(rows=rows), self.assertRaises(encounters.base.ProbeFailure):
                encounters.require_checks(rows, ["casualties"])

    def test_both_player_guard_fixtures_are_verified_against_actual_baseline(self):
        with TemporaryDirectory() as directory:
            for player, guard in (("red", "guard_0"), ("blue", "guard_1")):
                output = Path(directory) / player
                manifest = fixtures.create_fixture(output, guard=guard, count=20)
                target, checked = encounters.casualty_input(output / "fixture.json", player)
                self.assertEqual(str(target), manifest["fixture"]["path"])
                self.assertEqual(checked["change"], manifest["change"])
                with self.assertRaises(encounters.base.ProbeFailure):
                    encounters.casualty_input(output / "fixture.json", "blue" if player == "red" else "red")

    def test_rehashed_fixture_with_an_extra_gameplay_change_is_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            manifest = fixtures.create_fixture(output)
            target = Path(manifest["fixture"]["path"])
            with zipfile.ZipFile(target) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            objects = json.loads(entries["objects.json"])
            other = next(obj for obj in objects if obj["instanceName"] == "guard_1")
            other["options"]["amount"] += 1
            entries["objects.json"] = json.dumps(objects).encode()
            with zipfile.ZipFile(target, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
            manifest["fixture"]["sha256"] = encounters.base.sha(target)
            (output / "fixture.json").write_text(json.dumps(manifest))
            with self.assertRaises(encounters.base.ProbeFailure):
                encounters.casualty_input(output / "fixture.json", "red")

    def test_rehashed_fixture_boolean_cannot_replace_numeric_one(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            manifest = fixtures.create_fixture(output)
            target = Path(manifest["fixture"]["path"])
            with zipfile.ZipFile(target) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            objects = json.loads(entries["objects.json"])
            hero = next(obj for obj in objects if obj.get("options", {}).get("primarySkills", {}).get("attack") == 1)
            hero["options"]["primarySkills"]["attack"] = True
            entries["objects.json"] = json.dumps(objects).encode()
            with zipfile.ZipFile(target, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
            manifest["fixture"]["sha256"] = encounters.base.sha(target)
            (output / "fixture.json").write_text(json.dumps(manifest))
            with self.assertRaises(encounters.base.ProbeFailure):
                encounters.casualty_input(output / "fixture.json", "red")

    def test_resource_requires_actual_visit_credit_notice_removal_and_native_stop(self):
        args = resource_evidence()
        evidence = encounters.encounter_evidence("resource", *args)
        self.assertEqual(evidence["visit_seqs"], [3])
        self.assertEqual(evidence["notice_seqs"], [4])
        self.assertEqual(evidence["stop"]["cause_seq"], 3)
        self.assertEqual(evidence["stop"]["issued_after_cause"], [])
        self.assertEqual(evidence["gold_credit"], 500)

    def test_missing_or_unrelated_visit_or_ack_cannot_pass(self):
        for change in ("visit", "ack", "consumed", "wrong_ack_action", "wrong_ack_command"):
            target, before, after, frames, result = resource_evidence()
            if change in ("visit", "ack", "consumed"):
                index = {"visit": 2, "ack": 4, "consumed": 0}[change]
                del frames[index]
            elif change == "wrong_ack_action":
                frames[4] = replace(frames[4], decision_seq=1)
            else:
                frames[4] = replace(frames[4], command_index=1)
            with self.subTest(change=change), self.assertRaises(encounters.base.ProbeFailure):
                encounters.encounter_evidence("resource", target, before, after, frames, result)

    def test_unrelated_or_late_effect_does_not_satisfy_selected_action(self):
        for change in ("wrong_credit", "still_visible", "late_notice", "wrong_visit"):
            target, before, after, frames, result = resource_evidence()
            if change == "wrong_credit":
                after.resources = (0, 0, 0, 0, 0, 0, 601)
            elif change == "still_visible":
                after.objects = (target,)
            elif change == "late_notice":
                frames.append(replace(frames.pop(3), seq=7))
            else:
                frames[2] = replace(frames[2], event=VisibleEvent("visit", subject="another", object_kind=79))
            with self.subTest(change=change), self.assertRaises(encounters.base.ProbeFailure):
                encounters.encounter_evidence("resource", target, before, after, frames, result)

    def test_another_native_command_after_visit_fails_even_with_valid_effect(self):
        target, before, after, frames, result = resource_evidence()
        frames[-1] = replace(frames[-1], seq=8)
        frames[5:5] = [frame(6, "issued", decision_seq=0, command_index=1),
                       frame(7, "acknowledged", decision_seq=0, command_index=1, accepted=True)]
        with self.assertRaises(encounters.base.ProbeFailure):
            encounters.encounter_evidence("resource", target, before, after, frames, result)

    def test_battle_requires_completed_win_and_same_live_hero(self):
        target, before, after, frames, result = resource_evidence()
        target.kind = 54
        frames[2] = replace(frames[2], event=VisibleEvent("visit", subject=target.handle, object_kind=54))
        frames[3] = replace(frames[3], event=VisibleEvent("battle_start", side="attacker"))
        frames[-2:] = [frame(5, "event", event=VisibleEvent("battle_complete", won=True)),
                       replace(frames[-2], seq=6), replace(frames[-1], seq=7)]
        evidence = encounters.encounter_evidence("battle", target, before, after, frames, result)
        self.assertEqual(evidence["battle_seqs"], [4, 5])
        for won, heroes in ((False, before.heroes), (True, ()), (True, (SimpleNamespace(handle="replacement"),))):
            altered = [replace(f, event=VisibleEvent("battle_complete", won=won))
                       if f.seq == 5 else f for f in frames]
            after.heroes = heroes
            with self.subTest(won=won, heroes=heroes), self.assertRaises(encounters.base.ProbeFailure):
                encounters.encounter_evidence("battle", target, before, after, altered, result)

    def test_mine_requires_current_player_owned_capture(self):
        target, before, after, frames, result = resource_evidence()
        target.kind = 53
        frames[2] = replace(frames[2], event=VisibleEvent("visit", subject=target.handle, object_kind=53))
        after.objects = (SimpleNamespace(handle=target.handle, owner="self"),)
        encounters.encounter_evidence("mine", target, before, after, frames, result)
        after.objects[0].owner = "neutral"
        with self.assertRaises(encounters.base.ProbeFailure):
            encounters.encounter_evidence("mine", target, before, after, frames, result)


if __name__ == "__main__":
    unittest.main()
