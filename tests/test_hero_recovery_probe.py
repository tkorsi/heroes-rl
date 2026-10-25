"""Synthetic causal evidence guards, not real-game recovery validation."""
from dataclasses import replace
import unittest

from experiments.learner import contextual as c
from validation.probes.validate_contextual_native import ProbeFailure
from validation.probes.validate_hero_recovery import verified_loss


def frame(seq, kind, **payload):
    return c.Frame("synthetic-recovery", 0, seq, 1, kind, schema=c.HEROES_SCHEMA, **payload)


def evidence():
    original = c.Hero("original", (1, 1, 0), 100, 100)
    town = c.Town("retained", (0, 1, 0))
    before = c.Observation((4, 4, 1), (), (), (0, 0, 0, 0, 0, 0, 5000), (original,), (town,), None)
    after = replace(before, heroes=())
    start = frame(0, "decision", observation=before, revision=0,
                  offers=c.Offers.from_observation(before, schema=c.HEROES_ACTION_SCHEMA))
    battle = frame(1, "event", event=c.VisibleEvent("battle_start", side="attacker"))
    lost = frame(2, "event", event=c.VisibleEvent("hero_lost", subject=original.handle))
    defeat = frame(3, "event", event=c.VisibleEvent("battle_complete", won=False))
    current = frame(4, "decision", observation=after, revision=1,
                    offers=c.Offers.from_observation(after, schema=c.HEROES_ACTION_SCHEMA))
    return current, [start, battle, lost, defeat, current]


class HeroRecoveryProbeTests(unittest.TestCase):
    def test_actual_loss_with_retained_town_stays_ordinary(self):
        current, history = evidence()
        report = verified_loss(current, history, "original", "retained")
        self.assertFalse(report["episode_terminal"])
        self.assertEqual(report["hero_lost_seq"], 2)
        self.assertEqual(report["battle_defeat_seq"], 3)

    def test_empty_hero_list_without_visible_loss_or_defeat_is_not_proof(self):
        current, history = evidence()
        variants = ([history[0], history[1], history[3], current],
                    [history[0], history[1], history[2], current],
                    [history[0], history[2], history[3], current],
                    [history[1], history[2], history[3], current])
        for changed in variants:
            with self.subTest(history=changed), self.assertRaises(ProbeFailure):
                verified_loss(current, changed, "original", "retained")

    def test_other_hero_victory_or_unowned_town_cannot_prove_recovery(self):
        current, history = evidence()
        for changed in (
            history[:2] + [replace(history[2], event=c.VisibleEvent("hero_lost", subject="foreign"))] + history[3:],
            history[:3] + [replace(history[3], event=c.VisibleEvent("battle_complete", won=True)), current],
            [history[0], replace(history[1], event=c.VisibleEvent("battle_start", side="defender")), *history[2:]],
        ):
            with self.subTest(history=changed), self.assertRaises(ProbeFailure):
                verified_loss(current, changed, "original", "retained")
        no_town = replace(current.observation, towns=())
        with self.assertRaises(ProbeFailure):
            verified_loss(replace(current, observation=no_town,
                                  offers=c.Offers.from_observation(no_town, schema=c.HEROES_ACTION_SCHEMA)),
                          history, "original", "retained")
        still_alive = history[0].observation
        with self.assertRaises(ProbeFailure):
            verified_loss(replace(current, observation=still_alive,
                                  offers=c.Offers.from_observation(still_alive, schema=c.HEROES_ACTION_SCHEMA)),
                          history, "original", "retained")

    def test_episode_terminal_before_replacement_is_rejected(self):
        current, history = evidence()
        terminal = frame(3, "terminal", outcome="loss")
        with self.assertRaises(ProbeFailure):
            verified_loss(current, history + [terminal], "original", "retained")


if __name__ == "__main__":
    unittest.main()
