"""Synthetic finalizer-control regression; real Tavern cutoff is probed separately."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.headless.env import HeadlessSelfPlayEnv
from experiments.learner import contextual as c
from tests.test_contextual_heroes import observation, tavern


class Sent(Exception):
    pass


class HeroCutoffTests(unittest.TestCase):
    def test_open_tavern_finalizer_returns_to_overview_before_close(self):
        obs = observation(tavern())
        env = HeadlessSelfPlayEnv(SimpleNamespace(), Path('/synthetic-not-launched'), interface=c.HEROES_ENVIRONMENT)
        env.request = SimpleNamespace(player=0)
        env.reason = None
        env._settle = lambda: None
        env.sides = {
            0: SimpleNamespace(validator=SimpleNamespace(cutoff=object(), terminal=None,
                                                        live=SimpleNamespace(decision=object(), context=obs.context, requires_choice=False))),
            1: SimpleNamespace(validator=SimpleNamespace(cutoff=None, terminal=None)),
        }
        def send(action, *, continuation):
            self.assertTrue(continuation)
            self.assertEqual(action, c.Action('back', obs.context.subject, schema=c.HEROES_ACTION_SCHEMA))
            self.assertTrue(c.Offers.from_observation(obs, schema=c.HEROES_ACTION_SCHEMA).permits(action))
            raise Sent
        env._send_current = send
        with self.assertRaises(Sent): env._continue_cutoff()


if __name__ == '__main__': unittest.main()
