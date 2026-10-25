"""Synthetic mutation-generator tests; never substitute for native replay proof."""
from copy import deepcopy
import unittest

from experiments.headless.dialog_replay_mutations import adventure_dialogs_mutations
from experiments.headless.contextual_protocol import RECORD_FIELDS
from experiments.headless.replay import ReplayError, require_legacy_interface
from experiments.learner import contextual as c
from tests.test_contextual_dialogs import panel, history_at, begin_answer, finish_answer, open_dialog, observation
from tests.test_contextual_dialogs_notices import scholar, text


def tape_fixture(family="chest", *, chain=False, notices=False):
    history = history_at(panel(family))
    seq = begin_answer(history)
    if chain:
        second = panel("neutral_flee","dialog-2")
        history.emit("event",event=c.VisibleEvent("dialog_requested","hero",dialog=second))
        finish_answer(history,seq); open_dialog(history,second)
        seq = begin_answer(history)
    finish_answer(history,seq)
    if notices:
        for notice in (scholar(),
                       c.Notice("auto",(text("advevent",115),),(c.NoticeComponent("primary_skill",amount=1,skill="attack"),)),
                       c.Notice("auto",(text("advevent",145),c.NoticeToken("replace_text",text_id="artifact.core.centaurAxe.name")),
                                (c.NoticeComponent("artifact",content_id="core:centaurAxe"),))):
            history.emit("event",event=c.VisibleEvent("notice",notice=notice))
    profile = c.DIALOGS_PROFILE
    tape = [{"kind":"init","schema":"heroes-contextual-replay-v1","episode":history.episode,
             "interface":profile.environment,"protocol_version":profile.protocol,
             "history_schema":profile.history_schema,"action_schema":profile.action_schema}]
    for frame in history.frames:
        if frame.kind == "submitted":
            row = {"kind":"response","player":0,"response":{"decision_id":frame.decision_seq+100,
                   "action":c.encode_action(frame.action)}}
        else:
            record = {key:value for key,value in c.encode_frame(frame).items() if key in {"kind","day"} | RECORD_FIELDS[frame.kind]}
            if frame.kind == "decision": record["decision_id"] = frame.seq+100
            elif frame.decision_seq is not None: record["decision_id"] = frame.decision_seq+100
            row = {"kind":"message","player":0,"message":{"records":[record]}}
        tape.append(row)
    tape.insert(2,{"kind":"message","player":1,"message":{"records":[{"kind":"view","day":1,
                    "observation":c.encode_observation(observation(),schema=c.DIALOGS_SCHEMA)}]}})
    for index,row in enumerate(tape[1:]): row["seq"] = index
    return tape


def cases(tape):
    result = {}
    def add(name,index,mutate):
        assert name not in result
        changed = deepcopy(tape)
        mutate(changed,index)
        result[name] = changed
    adventure_dialogs_mutations(tape,add)
    return result


def records(tape):
    return [record for row in tape if row.get("kind") == "message" for record in row["message"].get("records",[])]


class DialogReplayMutationTests(unittest.TestCase):
    def test_partial_v7_metadata_cannot_enter_legacy_replay(self):
        for field, value in (("history_schema", c.DIALOGS_SCHEMA),
                             ("action_schema", c.DIALOGS_ACTION_SCHEMA)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ReplayError, "legacy digests"):
                    require_legacy_interface({field: value})
                with self.assertRaisesRegex(ReplayError, "legacy digests"):
                    require_legacy_interface({field: value, "protocol_version": 3})
        require_legacy_interface({"protocol_version": 3})

    def test_panel_callback_view_context_and_reply_mutations_are_independent(self):
        tape = tape_fixture(); original = deepcopy(tape)
        generated = cases(tape)
        for prefix in ("dialog_callback_","dialog_view_"):
            for suffix in ("reference","family","selection_flag","cancel_flag","text","component_amount",
                           "choice_reference","choice_kind","choice_component"):
                self.assertIn(prefix+suffix,generated)
        for name in ("dialog_callback_subject","dialog_view_context_reference","dialog_opened_reference",
                     "dialog_opened_subject","dialog_opened_revision","dialog_opened_reason","dialog_answer_choice",
                     "dialog_answer_subject","dialog_reply_acknowledgement","dialog_reply_result","dialog_reply_acceptance"):
            self.assertIn(name,generated)
        self.assertEqual(tape,original)
        self.assertEqual(generated,cases(tape))
        for name,changed in generated.items():
            self.assertNotEqual(changed,tape,name)
            self.assertEqual(changed[0],tape[0],name)
            self.assertEqual(changed[2],tape[2],name)  # Opponent does not provide current own dialog state.
            self.assertEqual([row["seq"] for row in changed[1:]],list(range(len(tape)-1)))
            self.assertEqual(sum(left != right for left,right in zip(changed,tape)),1,name)
        callback = lambda source: next(record["event"]["dialog"] for record in records(source)
                                      if record.get("event",{}).get("kind") == "dialog_requested")
        view = lambda source: next(record["observation"]["window"]["dialog_panel"] for record in records(source)
                                  if (record.get("observation",{}).get("window") or {}).get("dialog_panel"))
        self.assertEqual(callback(generated["dialog_view_component_amount"]),callback(tape))
        self.assertEqual(view(generated["dialog_callback_component_amount"]),view(tape))

    def test_recorded_selection_response_changes_to_another_actual_choice(self):
        tape = tape_fixture("school_magic")
        changed = cases(tape)["dialog_answer_choice"]
        response = next(row["response"] for row in changed if row.get("kind") == "response" and row["response"]["action"]["kind"] == "choose_dialog")
        selected = c.Action(**response["action"])
        support = history_at(panel("school_magic")).decision.offers
        self.assertTrue(support.permits(selected))
        original = next(row["response"] for row in tape if row.get("kind") == "response" and row["response"]["action"]["kind"] == "choose_dialog")
        self.assertNotEqual(response["action"]["selection"],original["action"]["selection"])
        self.assertIn("dialog_callback_component_skill",cases(tape))

    def test_stale_reference_requires_a_recorded_prior_same_player_dialog(self):
        self.assertNotIn("dialog_answer_stale_choice",cases(tape_fixture("neutral_free_join")))
        tape = tape_fixture("neutral_free_join",chain=True)
        changed = cases(tape)["dialog_answer_stale_choice"]
        changed_response = next(new for old,new in zip(tape,changed) if old != new)
        self.assertIn(changed_response["response"]["action"]["selection"],{choice.ref for choice in panel("neutral_free_join").choices})
        self.assertNotIn(changed_response["response"]["action"]["selection"],{choice.ref for choice in panel("neutral_flee","dialog-2").choices})

    def test_paid_and_free_payload_mutations_do_not_invent_hidden_counts(self):
        free = cases(tape_fixture("neutral_free_join"))
        self.assertIn("dialog_callback_creature_name",free)
        self.assertNotIn("dialog_callback_number",free)
        self.assertNotIn("dialog_callback_component_amount",free)
        paid = cases(tape_fixture("neutral_paid_join"))
        self.assertIn("dialog_callback_number",paid)
        self.assertIn("dialog_callback_component_amount",paid)

    def test_scholar_and_artifact_mutations_exist_only_for_recorded_notices(self):
        plain = cases(tape_fixture())
        self.assertFalse(any("scholar" in name or "artifact" in name or "notice" in name for name in plain))
        generated = cases(tape_fixture(notices=True))
        for name in ("dialog_notice_mode","dialog_scholar_mastery","dialog_scholar_skill","dialog_scholar_spell",
                     "dialog_scholar_separator","dialog_map_scholar_primary","dialog_map_scholar_amount","dialog_chest_artifact"):
            self.assertIn(name,generated)

    def test_exact_profile_binding_and_no_missing_coverage(self):
        tape = tape_fixture()
        for field,value in (("schema","unknown"),("interface",c.MANAGEMENT_ENVIRONMENT),("protocol_version",8),
                            ("protocol_version",9.0),("history_schema",c.MANAGEMENT_SCHEMA),("action_schema",c.MANAGEMENT_ACTION_SCHEMA)):
            changed = deepcopy(tape); changed[0][field] = value
            self.assertEqual(cases(changed),{},(field,value))
        self.assertEqual(cases([]),{})
        self.assertEqual(cases(tape[:3]),{})
        only_callback = [tape[0],next(row for row in tape if row.get("kind") == "message" and any(
            record.get("event",{}).get("kind") == "dialog_requested" for record in row["message"]["records"]))]
        self.assertTrue(cases(only_callback))
        self.assertTrue(all(name.startswith("dialog_callback_") for name in cases(only_callback)))

    def test_reply_evidence_binds_matching_player_and_decision(self):
        for mismatch in ("player","decision_id","superseded"):
            tape = tape_fixture()
            for row in tape:
                for record in row.get("message",{}).get("records",[]):
                    if record.get("kind") == "acknowledged" and record.get("decision_id",0) > 100:
                        if mismatch == "player": row["player"] = 1
                        else: record["decision_id"] = 999
                    if record.get("kind") == "result" and record.get("result",{}).get("status") == "dialog_answered":
                        if mismatch == "player": row["player"] = 1
                        elif mismatch == "decision_id": record["decision_id"] = 999
                        else:
                            record.pop("result"); record.update(kind="superseded",reason="terminal")
            generated = cases(tape)
            self.assertNotIn("dialog_reply_acknowledgement",generated)
            self.assertNotIn("dialog_reply_result",generated)
        tape = tape_fixture()
        for row in tape:
            for record in row.get("message",{}).get("records",[]):
                if (record.get("observation",{}).get("window") or {}).get("dialog_panel"):
                    row["player"] = 1
        generated = cases(tape)
        self.assertNotIn("dialog_answer_choice",generated)
        self.assertNotIn("dialog_reply_result",generated)


if __name__ == "__main__": unittest.main()
