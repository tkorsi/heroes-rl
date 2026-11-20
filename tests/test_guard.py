"""The guard's daily budgets, set through its name. They change what a policy may sample."""
import pytest

from heroes.game import contract as c
from heroes.game import guard

SCHEMA = c.profile_for(c.EXPANDED_ENVIRONMENT)
TILES = tuple(c.Tile((x, y, 0), 1, 0, False, False) for y in range(3) for x in range(3))
MAP = c.Observation((3, 3, 1), TILES, (), (0,) * 7,
                    (c.Hero("object-0", (0, 0, 0), 900, 1500, False), c.Hero("object-1", (2, 2, 0), 900, 1500, False)),
                    (c.Town("object-2", (1, 1, 0)),), army_information=c.OWN_ARMY_INFORMATION,
                    own_armies=tuple(c.OwnArmy(subject, role, ()) for subject, role in (
                        ("object-0", "hero"), ("object-1", "hero"), ("object-2", "town_garrison"))))
OFFERS = c.Offers.from_observation(MAP, schema=SCHEMA.action_schema)


def action(kind, subject=None):
    return c.Action(kind, subject, schema=SCHEMA.action_schema)


class Seat:
    """A tracker fed the frames of a seat that is asked on the map and answers."""

    def __init__(self, name):
        self.tracker, self.frames, self.day = guard.Tracker(name), 0, 1

    def frame(self, kind, **payload):
        made = c.Frame("1", 0, self.frames, self.day, kind, schema=SCHEMA.history_schema, **payload)
        self.frames += 1
        return self.tracker.observe(made)

    def decide(self):
        return self.frame("decision", observation=MAP, offers=OFFERS, revision=self.frames)

    def answer(self, chosen):
        decision = self.frames - 1
        self.frame("submitted", action=chosen, decision_seq=decision)
        self.frame("consumed", consumer="engine", decision_seq=decision)


def test_the_standard_guard_keeps_its_name_and_other_budgets_are_part_of_the_name():
    assert guard.version() == guard.version(stack_moves=32) == guard.VERSION
    assert guard.budgets(guard.VERSION) == (32, None, None)
    name = guard.version(stack_moves=8, windows=6, decisions=40)
    assert name == guard.VERSION + "+stack-moves-8+windows-6+decisions-40"
    assert guard.budgets(name) == (8, 6, 40) and guard.budgets(guard.version(decisions=40)) == (32, None, 40)
    assert guard.Tracker(name).management_limit == 8 and guard.Tracker(None).version is None
    for bad in ("other-guard-v1", guard.VERSION + "+stack-moves-32", guard.VERSION + "+windows-0",
                guard.VERSION + "+decisions-40+windows-6", guard.VERSION + "+moves-3", guard.VERSION + "+windows-x",
                guard.VERSION + "+windows-6+windows-6"):
        with pytest.raises(ValueError, match="Unsupported interaction guard"):
            guard.Tracker(bad)
    with pytest.raises(ValueError):
        guard.version(windows=0)


def test_a_window_budget_closes_a_hero_for_the_day_and_no_other():
    seat = Seat(guard.version(windows=1))
    support = seat.decide()
    assert support.permits(action("open_hero", "object-0"), MAP, OFFERS) and not support.blocked_window_subjects
    seat.answer(action("open_hero", "object-0"))
    support = seat.decide()
    assert support.blocked_window_subjects == ("object-0",)
    assert not support.permits(action("open_hero", "object-0"), MAP, OFFERS)
    assert support.permits(action("open_hero", "object-1"), MAP, OFFERS)
    assert support.permits(action("open_town", "object-2"), MAP, OFFERS)
    assert support.permits(action("end_turn"), MAP, OFFERS)
    seat.answer(action("open_town", "object-2"))
    support = seat.decide()
    assert support.blocked_window_subjects == ("object-0", "object-2")
    assert not support.permits(action("open_hero", "object-0"), MAP, OFFERS)
    # The standard guard lets the hero be opened a second time, only not twice in a row.
    standard = Seat(guard.VERSION)
    for chosen in (action("open_hero", "object-0"), action("open_town", "object-2")):
        standard.decide()
        standard.answer(chosen)
    assert standard.decide().permits(action("open_hero", "object-0"), MAP, OFFERS)
    seat.day = 2
    assert seat.decide().permits(action("open_hero", "object-0"), MAP, OFFERS)


def test_past_the_decision_budget_a_seat_can_only_end_its_turn_until_the_next_day():
    seat = Seat(guard.version(decisions=2))
    for subject in ("object-0", "object-1"):
        support = seat.decide()
        assert not support.exhausted and support.permits(action("open_town", "object-2"), MAP, OFFERS)
        seat.answer(action("open_hero", subject))
    support = seat.decide()
    assert support.exhausted
    assert not support.permits(action("open_town", "object-2"), MAP, OFFERS)
    assert not support.permits(action("open_hero", "object-1"), MAP, OFFERS)
    assert support.permits(action("end_turn"), MAP, OFFERS)
    seat.answer(action("end_turn"))
    seat.day = 2
    support = seat.decide()
    assert not support.exhausted and support.permits(action("open_town", "object-2"), MAP, OFFERS)
