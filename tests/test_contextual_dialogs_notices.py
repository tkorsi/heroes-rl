"""Exact v7 normalized informational payloads, separate from strategic choices."""
from copy import deepcopy
from dataclasses import replace
import unittest

from experiments.learner import contextual as c


def text(namespace, number):
    return c.NoticeToken("append_text", text_id=f"core.{namespace}.{number}")


def scholar(learned=("bless", "haste", "shield"), taught=("slow",)):
    tokens = [text("genrltxt",139), c.NoticeToken("replace_text",text_id="hero.core.catherine.name")]
    components = [c.NoticeComponent("secondary_skill",amount=3,content_id="core:scholar")]
    for index, spells in enumerate((learned,taught)):
        if not spells: continue
        if index and learned: tokens.append(text("genrltxt",141))
        tokens.append(text("genrltxt",147 if index else 140))
        for position, spell in enumerate(spells):
            if position: tokens.append(text("genrltxt",141) if position == len(spells)-1 else
                                       c.NoticeToken("append_text",text_id="heroes.notice.comma_space"))
            tokens.append(c.NoticeToken("append_text",text_id=f"spell.core.{spell}.name"))
            components.append(c.NoticeComponent("spell",content_id=f"core:{spell}"))
        tokens += [text("genrltxt",148 if index else 142), c.NoticeToken("replace_text",text_id="hero.core.adela.name")]
    return c.Notice("modal",tuple(tokens),tuple(components))


class DialogNoticeTests(unittest.TestCase):
    def roundtrip(self, notice):
        raw = c._wire(notice)
        self.assertEqual(c.decode_notice(raw,schema=c.DIALOGS_SCHEMA),notice)
        for profile in (c.WINDOWS_PROFILE,c.TOWN_PROFILE,c.ARMIES_PROFILE,c.HEROES_PROFILE,c.MANAGEMENT_PROFILE):
            with self.assertRaises(ValueError): c.decode_notice(raw,schema=profile.history_schema)

    def test_automatic_map_scholar_exposes_only_actual_component(self):
        for component in (c.NoticeComponent("primary_skill",amount=1,skill="attack"),
                          c.NoticeComponent("secondary_skill",amount=2,content_id="core:logistics"),
                          c.NoticeComponent("spell",content_id="core:haste")):
            notice = c.Notice("auto",(text("advevent",115),),(component,))
            self.roundtrip(notice)
            for extra in (c.NoticeToken("replace_text",text_id="spell.core.slow.name"),
                          c.NoticeToken("replace_text",text_id="skill.core.learning.name")):
                with self.assertRaises(ValueError): c.decode_notice(c._wire(replace(notice,text=notice.text+(extra,))),schema=c.DIALOGS_SCHEMA)
            with self.assertRaises(ValueError): c.decode_notice(c._wire(replace(notice,components=(component,component))),schema=c.DIALOGS_SCHEMA)
        with self.assertRaises(ValueError): c.NoticeComponent("primary_skill",amount=True,skill="attack")

    def test_artifact_chest_retains_only_rendered_artifact_name(self):
        notice = c.Notice("auto",(text("advevent",145),c.NoticeToken("replace_text",text_id="artifact.core.centaurAxe.name")),
                          (c.NoticeComponent("artifact",content_id="core:centaurAxe"),))
        self.roundtrip(notice)
        wrong = replace(notice,components=(c.NoticeComponent("artifact",content_id="core:greaterGnollsFlail"),))
        with self.assertRaises(ValueError): c.decode_notice(c._wire(wrong),schema=c.DIALOGS_SCHEMA)
        raw = c._wire(notice); raw["text"].append(dict(op="replace_raw",text="Treasure Chest"))
        with self.assertRaises(ValueError): c.decode_notice(raw,schema=c.DIALOGS_SCHEMA)

    def test_revisit_insufficient_resources_and_neutral_refusal_notices(self):
        for number in (1,72,73,85,159,160):
            self.roundtrip(c.Notice("auto",(text("advevent",number),),()))
        for namespace,number in (("advevent",87),("genrltxt",29)):
            notice = c.Notice("modal",(text(namespace,number),),())
            self.roundtrip(notice)
            with self.assertRaises(ValueError): c.decode_notice(c._wire(replace(notice,display_mode="auto")),schema=c.DIALOGS_SCHEMA)
        with self.assertRaises(ValueError): c.decode_notice(c._wire(c.Notice("modal",(text("advevent",86),),())),schema=c.DIALOGS_SCHEMA)

    def test_scholar_lists_bind_each_displayed_spell_in_order(self):
        for learned,taught in ((("haste",),()),((),("slow",)),(("haste","shield"),()),
                              (("bless","haste","shield"),("slow",))):
            self.roundtrip(scholar(learned,taught))
        notice = scholar()
        changes = (replace(notice,components=(notice.components[0],*reversed(notice.components[1:]))),
                   replace(notice,text=tuple(item for item in notice.text if item.text_id != "heroes.notice.comma_space")),
                   replace(notice,display_mode="auto"),
                   replace(notice,components=(replace(notice.components[0],content_id="core:wisdom"),*notice.components[1:])))
        for changed in changes:
            with self.assertRaises(ValueError): c.decode_notice(c._wire(changed),schema=c.DIALOGS_SCHEMA)
        raw = c._wire(notice)
        raw["text"][1]["text_id"] = "hidden.enemy.current.position"
        with self.assertRaises(ValueError): c.decode_notice(raw,schema=c.DIALOGS_SCHEMA)

    def test_large_scholar_list_has_v7_only_bounds_and_no_generic_raw_text(self):
        notice = scholar(tuple(f"spell{index}" for index in range(40)),())
        self.roundtrip(notice)
        self.assertGreater(len(notice.components),32)
        with self.assertRaises(ValueError): c.Notice("auto",(text("advevent",115),)*65,())
        raw = c._wire(scholar()); raw["text"].insert(3,dict(op="append_raw",text=", "))
        with self.assertRaises(ValueError): c.decode_notice(raw,schema=c.DIALOGS_SCHEMA)
        comma = c.Notice("auto",(c.NoticeToken("append_text",text_id="heroes.notice.comma_space"),),())
        with self.assertRaises(ValueError): c.decode_notice(c._wire(comma),schema=c.DIALOGS_SCHEMA)


if __name__ == "__main__": unittest.main()
