"""Rules of the game contract that are computed in more than one way."""
import random

from heroes.game import contract as c


def _army(rng, index, role):
    nonempty = role not in ("town_garrison", "joining_neutral", *c.OBJECT_GARRISON_ROLES)
    filled = [rng.random() < 0.5 for _ in range(7)]
    if nonempty and not any(filled):
        filled[rng.randrange(7)] = True
    slots = tuple(c.ArmySlot(f"slot-{index}-{slot}", slot, rng.randrange(1, 4) if full else None,
                             rng.choice((1, 2, 30, 2**31 - 1)) if full else 0)
                  for slot, full in enumerate(filled))
    return c.ArmyView(f"army-{index}", role, f"subject-{index}", nonempty, slots)


def test_listing_stack_operations_agrees_with_asking_about_every_pair():
    rng = random.Random(1)
    roles = ("hero", "visiting_hero", "garrison_hero", "town_garrison", "joining_neutral", *c.OBJECT_GARRISON_ROLES)
    listed = 0
    for _ in range(150):
        armies = tuple(_army(rng, index, role) for index, role in enumerate(rng.sample(roles, rng.choice((1, 2)))))
        for kind in ("split_stack", "merge_stacks", "swap_stacks"):
            expected = tuple((source.ref, destination.ref, maximum)
                             for first in armies for source in first.slots
                             for second in armies for destination in second.slots
                             if (maximum := c.army_operation_maximum(armies, kind, source.ref, destination.ref)) > 0)
            assert c.army_operation_choices(armies, kind) == expected
            assert c.has_army_operation(armies, kind) == bool(expected)
            listed += len(expected)
    assert listed > 1000  # the layouts do offer operations of every kind
