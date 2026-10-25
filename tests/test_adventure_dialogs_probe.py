"""Adversarial dialog-probe oracles and failure bounds; no native games."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from experiments.learner import contextual as c
from validation.probes import validate_adventure_dialogs as p


def mine_panel():
    return c.DialogPanel("dialog-0", "guarded_mine", (c.NoticeToken("append_text", text_id="core.advevent.187"),),
        (), False, True, (c.DialogChoice("yes", "accept"), c.DialogChoice("no", "decline")))


class AdventureDialogProbeTests(unittest.TestCase):
    def test_choice_ref_follows_displayed_kind_not_ordinal(self):
        panel = mine_panel()
        self.assertEqual(p.explicit_choice(panel, "decline").ref, "no")
        with self.assertRaisesRegex(p.base.ProbeFailure, "uniquely displayed"):
            p.explicit_choice(panel, "cancel")
        reordered = NS(choices=tuple(reversed(panel.choices)))
        self.assertEqual(p.explicit_choice(reordered, "accept").ref, "yes")

    def test_opening_requires_same_payload_and_parent_query_order(self):
        panel = mine_panel(); context = c.Context("dialog", "hero", selection=panel.ref)
        parent = NS(seq=10)
        result = c.Result("interaction", True)
        after = NS(seq=15, observation=NS(context=context,
            window=NS(dialog_panel=panel, armies=(), army=None, stack_panel=None)))
        event = NS(kind="event", seq=11, event=NS(kind="dialog_requested", dialog=panel, subject="hero"))
        end = NS(kind="result", seq=12, event=None, decision_seq=10, result=result)
        opened = NS(kind="context_opened", seq=14, event=None, context=context, reason="adventure_dialog")
        self.assertEqual(p.dialog_open_proof(parent, after, (event,end,opened), result)["family"], "guarded_mine")
        event.seq=13
        with self.assertRaisesRegex(p.base.ProbeFailure, "reordered"):
            p.dialog_open_proof(parent, after, (event,end,opened), result)
        event.seq=11; event.event.subject="enemy"
        with self.assertRaises(p.base.ProbeFailure): p.dialog_open_proof(parent, after, (event,end,opened), result)

    def test_gold_effect_checks_all_seven_resources(self):
        before=(10,2,10,2,2,2,5000)
        self.assertEqual(p.exact_resource_delta(before,(10,2,10,2,2,2,4000),-1000),[0,0,0,0,0,0,-1000])
        with self.assertRaisesRegex(p.base.ProbeFailure, "beyond"):
            p.exact_resource_delta(before,(9,2,10,2,2,2,4000),-1000)

    def test_join_effect_uses_declared_fixture_count_only_in_oracle(self):
        before=NS(army=(NS(creature=0,count=1500), NS(creature=2,count=7)))
        after=NS(army=(NS(creature=0,count=1680), NS(creature=2,count=7)))
        p.joined_army_proof(before,after,180)
        with self.assertRaisesRegex(p.base.ProbeFailure,"fixture army effect"):
            p.joined_army_proof(before,after,6)
        after.army=(NS(creature=0,count=1680), NS(creature=2,count=8))
        with self.assertRaises(p.base.ProbeFailure): p.joined_army_proof(before,after,180)

    def test_unaffordable_acceptance_requires_real_price_notice_and_chained_query(self):
        paid=NS(family="neutral_paid_join",components=(NS(kind="resource",amount=10800),))
        flee=NS(family="neutral_flee",ref="dialog-1")
        resources=(5,2,5,2,2,2,5900)
        before=NS(observation=NS(resources=resources,window=NS(dialog_panel=paid)))
        after=NS(observation=NS(resources=resources,context=NS(kind="dialog"),window=NS(dialog_panel=flee)))
        notice=NS(seq=20,event=NS(kind="notice",notice=NS(text=(c.NoticeToken("append_text",text_id="core.genrltxt.29"),))))
        request=NS(seq=21,event=NS(kind="dialog_requested",dialog=flee))
        self.assertEqual(p.unaffordable_paid_proof(before,after,(notice,request))["available_gold"],5900)
        with self.assertRaisesRegex(p.base.ProbeFailure,"GENERAL29"):
            p.unaffordable_paid_proof(before,after,(request,))
        request.seq=19
        with self.assertRaisesRegex(p.base.ProbeFailure,"ordered"):
            p.unaffordable_paid_proof(before,after,(notice,request))
        request.seq=21;after.observation.resources=(4,2,5,2,2,2,5900)
        with self.assertRaisesRegex(p.base.ProbeFailure,"beyond"):
            p.unaffordable_paid_proof(before,after,(notice,request))
        after.observation.resources=resources;paid.components[0].amount=360
        with self.assertRaisesRegex(p.base.ProbeFailure,"affordable"):
            p.unaffordable_paid_proof(before,after,(notice,request))

    def test_refusal_attack_requires_actual_notice_and_completed_battle_order(self):
        notice=NS(seq=20,event=NS(kind="notice",notice=NS(text=(c.NoticeToken("append_text",text_id="core.advevent.87"),))))
        start=NS(seq=21,event=NS(kind="battle_start"))
        end=NS(seq=23,event=NS(kind="battle_complete"))
        self.assertEqual(p.refusal_attack_proof((notice,start,end))["refusal_notice_seq"],20)
        with self.assertRaisesRegex(p.base.ProbeFailure,"ADVOB87"):
            p.refusal_attack_proof((start,end))
        end.seq=19
        with self.assertRaisesRegex(p.base.ProbeFailure,"ordered"):
            p.refusal_attack_proof((notice,start,end))

    def test_exhausted_mandatory_dialog_never_becomes_a_cutoff_answer(self):
        run=object.__new__(p.DialogRun)
        run.state=NS(done=False); run.commands=2
        run.args=NS(max_decisions=2,case_seconds=120)
        run.started=p.time.monotonic()
        run.inspect_request=Mock(return_value=NS(observation=NS(context=NS(kind="dialog"))))
        run.env=NS(truncate=Mock())
        with self.assertRaisesRegex(p.base.ProbeFailure, "mandatory dialog"):
            run.live()
        run.env.truncate.assert_not_called()

    def test_never_uses_unseen_fixture_coordinate_for_movement(self):
        run=object.__new__(p.DialogRun)
        run.live=Mock(return_value=True)
        run.inspect_request=Mock(return_value=NS(observation=NS(context=NS(kind="map"),objects=())))
        run.call=Mock()
        with self.assertRaisesRegex(p.base.ProbeFailure, "currently visible"):
            run.visit(NS(handle="hidden-object",position=(10,10,0)))
        run.call.assert_not_called()


if __name__ == "__main__": unittest.main()
