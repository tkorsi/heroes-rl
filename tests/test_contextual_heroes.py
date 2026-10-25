"""Synthetic tavern contract checks; these are not native gameplay evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from experiments.learner import contextual as c
from tests.test_contextual_armies import army


def tavern(*, selected="offer-a", status="allowed", empty=False):
    offers = () if empty else (c.HeroRecruitChoice("offer-a", 0, "hero.core.orrin.name"),
                               c.HeroRecruitChoice("offer-b", 1, "hero.core.valeska.name"))
    return c.TavernPanel(offers, None if empty else selected,
                        None if empty else c.HeroRecruitDetails(1, int(selected == "offer-b"), 0),
                        2500, "no_offer" if empty else status)


def observation(panel=None, *, heroes=None, gold=7000):
    own = (c.Hero("hero", (0, 0, 0), 100, 100),) if heroes is None else heroes
    window = None
    if panel == "overview":
        upper = army("upper", "town_garrison", "town", {})
        payload = c.TownPanel("overview", (c.PanelControl("hall", "hall"), c.PanelControl("tavern", "tavern")),
                              armies=c.ArmyExchange(upper, None))
        window = c.Window(c.Context("town", "town", "overview"), (), faction=0, town_panel=payload)
    elif panel is not None:
        window = c.Window(c.Context("town", "town", "tavern"), None,
                          town_panel=c.TownPanel("tavern", tavern=panel))
    return c.Observation((3, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (),
                         (0, 0, 0, 0, 0, 0, gold), own, (c.Town("town", (1, 1, 0)),), window)


def action(kind, **kwargs):
    return c.Action(kind, schema=c.HEROES_ACTION_SCHEMA, **kwargs)


def decide(history, obs):
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=c.HEROES_ACTION_SCHEMA),
                        revision=history.revision + 1)


def navigate(history, selected, obs, status="opened"):
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=selected)
    history.emit("consumed", decision_seq=seq, consumer="adapter")
    history.emit("context", decision_seq=seq, context=obs.context, revision=history.revision + 1)
    history.emit("view", observation=obs)
    history.emit("result", decision_seq=seq, result=c.Result(status, True))
    return decide(history, obs)


def history_at_tavern(player=0, *, payload=None, heroes=None, gold=7000):
    history = c.PlayerHistory("synthetic-heroes", player, schema=c.HEROES_SCHEMA)
    decide(history, observation(heroes=heroes, gold=gold))
    navigate(history, action("open_town", subject="town"), observation("overview", heroes=heroes, gold=gold))
    navigate(history, action("open_panel", subject="town", panel="tavern"),
             observation(payload or tavern(), heroes=heroes, gold=gold))
    return history


def begin_hire(history, *, accepted=True):
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=action("hire_hero", subject="town",
                 selection=history.decision.observation.window.town_panel.tavern.selected_offer_ref))
    history.emit("consumed", decision_seq=seq, consumer="engine")
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("context", decision_seq=seq, context=c.Context("town", "town", "overview"), revision=history.revision + 1)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=accepted)
    return seq


def hired_observation(before, *, accepted=True):
    heroes = before.heroes + ((c.Hero("new-hero", (1, 1, 0), 100, 100),) if accepted else ())
    return observation("overview", heroes=heroes,
                       gold=before.resources[6] - (before.window.town_panel.tavern.gold_cost if accepted else 0))


class HeroContractTests(unittest.TestCase):
    def test_v5_is_explicit_and_old_profiles_reject_tavern(self):
        self.assertEqual(c.profile_for(c.HEROES_ENVIRONMENT), c.HEROES_PROFILE)
        self.assertEqual((c.HEROES_PROFILE.protocol, c.HEROES_PROFILE.version), (7, 5))
        frame = history_at_tavern().decision
        raw = c.encode_frame(frame)
        self.assertEqual(c.decode_frame(json.loads(json.dumps(raw))), frame)
        for profile in (c.WINDOWS_PROFILE, c.TOWN_PROFILE, c.ARMIES_PROFILE):
            changed = deepcopy(raw); changed["schema"] = profile.history_schema
            with self.assertRaises(ValueError): c.decode_frame(changed)
            with self.assertRaises(ValueError):
                c.Action("hire_hero", "town", selection="offer-a", schema=profile.action_schema)
            with self.assertRaises(ValueError):
                c.Action("open_panel", "town", panel="tavern", schema=profile.action_schema)
            with self.assertRaises(ValueError): c.encode_observation(observation("overview"), schema=profile.history_schema)

    def test_tavern_payload_rejects_hidden_details_and_keeps_old_snapshot_immutable(self):
        frame = history_at_tavern().decision
        with self.assertRaises(FrozenInstanceError): frame.observation.window.town_panel.tavern.gold_cost = 1
        for field in ("army", "primary_skills", "current_position", "hero_instance_id"):
            changed = c.encode_frame(frame)
            changed["observation"]["window"]["town_panel"]["tavern"]["offers"][0][field] = []
            with self.assertRaises(ValueError): c.decode_frame(changed)
        with self.assertRaises(ValueError): c.TownPanel("overview", tavern=tavern())
        with self.assertRaises(ValueError): c.Window(c.Context("town", "town", "tavern"), (),
            town_panel=c.TownPanel("tavern", tavern=tavern()))
        with self.assertRaises(ValueError): replace(tavern(), selected_offer_ref="unoffered")
        with self.assertRaises(ValueError): replace(tavern(empty=True), selected_detail=c.HeroRecruitDetails(1, 0, 0))

    def test_only_selected_available_card_can_be_hired_without_invented_hero_cap(self):
        history = history_at_tavern(heroes=tuple(c.Hero(f"hero-{n}", (0, 0, 0), 0, 100) for n in range(17)))
        offers = history.decision.offers
        self.assertTrue(offers.permits(action("hire_hero", subject="town", selection="offer-a")))
        self.assertFalse(offers.permits(action("hire_hero", subject="town", selection="offer-b")))
        self.assertFalse(offers.permits(action("hire_hero", subject="foreign", selection="offer-a")))
        for selected in ("offer-a", "offer-b"):
            self.assertTrue(offers.permits(action("select_hero", subject="town", selection=selected)))
        for status in ("no_gold", "total_limit", "map_limit", "visitor_present"):
            blocked = history_at_tavern(payload=tavern(status=status), gold=100 if status == "no_gold" else 7000)
            self.assertFalse(blocked.decision.offers.permits(action("hire_hero", subject="town", selection="offer-a")))
            self.assertTrue(blocked.decision.offers.permits(action("select_hero", subject="town", selection="offer-b")))
        empty = history_at_tavern(payload=tavern(empty=True))
        self.assertFalse(empty.decision.offers.permits(action("select_hero", subject="town", selection="offer-a")))
        self.assertTrue(empty.decision.offers.permits(action("back", subject="town")))
        # Native UI checks gold/caps/visitor before whether a card exists.
        self.assertEqual(replace(tavern(empty=True), recruit_status="no_gold").recruit_status, "no_gold")

    def test_cost_status_uses_visible_gold_and_no_other_resource_debit(self):
        with self.assertRaises(ValueError): observation(tavern(), gold=2499)
        with self.assertRaises(ValueError): observation(tavern(status="no_gold"), gold=2500)
        for cost in (-1, True, 2**31):
            with self.assertRaises(ValueError): replace(tavern(), gold_cost=cost)

    def test_selected_details_require_explicit_revision_and_prior_snapshot_does_not_change(self):
        history = history_at_tavern(); original = history.decision
        raw = c.encode_frame(original)
        navigate(history, action("select_hero", subject="town", selection="offer-b"),
                 observation(tavern(selected="offer-b")), status="selected")
        self.assertEqual(c.encode_frame(original), raw)
        self.assertGreater(history.decision.revision, original.revision)
        self.assertTrue(history.decision.offers.permits(action("hire_hero", subject="town", selection="offer-b")))
        self.assertFalse(history.decision.offers.permits(action("hire_hero", subject="town", selection="offer-a")))
        self.assertFalse(any(frame.kind in ("issued", "acknowledged") for frame in history.frames))

    def test_selecting_card_cannot_change_resources_or_other_cards(self):
        for corruption in ("gold", "cards", "cost", "status"):
            history = history_at_tavern(); seq = history.decision.seq
            history.emit("submitted", decision_seq=seq, action=action("select_hero", subject="town", selection="offer-b"))
            history.emit("consumed", decision_seq=seq, consumer="adapter")
            history.emit("context", decision_seq=seq, context=history.context, revision=history.revision + 1)
            selected = tavern(selected="offer-b")
            if corruption == "cards": selected = replace(selected, offers=(replace(selected.offers[0], portrait=99), selected.offers[1]))
            elif corruption == "cost": selected = replace(selected, gold_cost=2501)
            elif corruption == "status": selected = replace(selected, recruit_status="visitor_present")
            history.emit("view", observation=observation(selected, gold=6999 if corruption == "gold" else 7000))
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                history.emit("result", decision_seq=seq, result=c.Result("selected", True))

    def test_successful_hire_and_recovery_require_one_packet_exact_cost_and_one_new_hero(self):
        for player in (0, 1):
            for heroes in ((), (c.Hero("hero", (0, 0, 0), 100, 100),)):
                history = history_at_tavern(player, heroes=heroes)
                before = history.decision.observation
                seq = begin_hire(history)
                after = hired_observation(before)
                history.emit("view", observation=after)
                history.emit("result", decision_seq=seq, result=c.Result("hired", True))
                decide(history, after)
                navigate(history, action("close"), replace(after, window=None), status="closed")
                self.assertEqual(history.decision.offers.hero_refs, tuple(hero.handle for hero in after.heroes))
                self.assertTrue(history.decision.offers.permits(action("move_to", subject="new-hero", x=2, y=0)))
                self.assertEqual(before.heroes, heroes)
                self.assertIsNotNone(before.window.town_panel.tavern)
                self.assertIsNone(history.decision.observation.window)

    def test_missing_receipt_wrong_gold_or_hero_creation_rejected(self):
        for corrupt in ("no_new", "two_new", "old_lost", "old_changed", "wrong_gold", "other_resource"):
            history = history_at_tavern(); before = history.decision.observation
            seq = begin_hire(history); after = hired_observation(before)
            if corrupt == "no_new": after = replace(after, heroes=before.heroes)
            elif corrupt == "two_new": after = replace(after, heroes=after.heroes + (c.Hero("extra", (0, 0, 0), 100),))
            elif corrupt == "old_lost": after = replace(after, heroes=after.heroes[1:])
            elif corrupt == "old_changed": after = replace(after, heroes=(replace(after.heroes[0], movement=0), after.heroes[1]))
            elif corrupt == "wrong_gold": after = replace(after, resources=(0, 0, 0, 0, 0, 0, 5000))
            else: after = replace(after, resources=(1, 0, 0, 0, 0, 0, 4500))
            history.emit("view", observation=after)
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                history.emit("result", decision_seq=seq, result=c.Result("hired", True))
        history = history_at_tavern(); seq = history.decision.seq
        history.emit("submitted", decision_seq=seq, action=action("hire_hero", subject="town", selection="offer-a"))
        with self.assertRaises(ValueError): history.emit("consumed", decision_seq=seq, consumer="adapter")
        history.emit("consumed", decision_seq=seq, consumer="engine")
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("hired", True))

    def test_rejection_keeps_resources_and_existing_heroes_and_back_removes_private_details(self):
        history = history_at_tavern(); before = history.decision.observation
        seq = begin_hire(history, accepted=False)
        after = hired_observation(before, accepted=False)
        history.emit("view", observation=after)
        history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
        decide(history, after)
        self.assertIsNone(history.decision.observation.window.town_panel.tavern)
        self.assertFalse(history.decision.offers.permits(action("hire_hero", subject="town", selection="offer-a")))
        history = history_at_tavern()
        navigate(history, action("back", subject="town"), observation("overview"), status="closed")
        self.assertNotIn('"gold_cost"', json.dumps(c.encode_frame(history.decision)))


if __name__ == "__main__": unittest.main()
