"""Acting reuses what it knows of the unchanged parts of an observation; nothing it computes may change.

The scripted engine shows no armies, so these observations are built by hand:
two own heroes exchanging stacks, a town, and the passive statistics of every
owned army. One army changes from view to view while the rest stays.
"""
import pytest
import torch

from heroes.game import contract as c
from heroes.game import guard
from heroes.learner.recipes import compile_history
from heroes.model import policy
from heroes.model.policy import ContextualActorCritic, encode_history

SCHEMA = c.profile_for(c.EXPANDED_ENVIRONMENT)
TILES = tuple(c.Tile((x, y, 0), 1 + (x + y) % 3, 0, False, False) for y in range(4) for x in range(4))
CREATURES = {0: "pikeman", 2: "archer", 4: "griffin", 6: "swordsman"}


def own_army(subject, role, layout):
    return c.OwnArmy(subject, role, tuple(
        c.OwnArmyStack(slot, creature, count, f"creatures.core.{CREATURES[creature]}.name.plural", 4 + creature, 5, 1, 3,
                       10 + creature, 4, 0, 0, 1, 0, False, creature == 4, False)
        for slot, (creature, count) in enumerate(layout) if count))


def shown_army(index, subject, layout):
    return c.ArmyView(f"army-{index}", "hero", subject, True, tuple(
        c.ArmySlot(f"army-{index}-slot-{slot}", slot, creature if count else None, count)
        for slot, (creature, count) in enumerate(layout)))


def observation(first, second, gold):
    """Hero object-0 exchanging with hero object-1; ``first`` and ``second`` are their seven (creature, count) slots."""
    heroes = (c.Hero("object-0", (1, 1, 0), 900, 1500, False), c.Hero("object-1", (2, 1, 0), 0, 1500, False))
    window = c.Window(c.Context("exchange", "object-0", None, "object-1"), None,
                      armies=(shown_army(0, "object-0", first), shown_army(1, "object-1", second)),
                      capability_controls=())
    typed = c.Observation((4, 4, 1), TILES, (c.ObjectView("object-3", (3, 3, 0), 53, 0, "neutral"),),
                          (1, 2, 3, 4, 5, 6, gold), heroes, (c.Town("object-2", (0, 3, 0)),), window,
                          army_information=c.OWN_ARMY_INFORMATION,
                          own_armies=(own_army("object-0", "hero", first), own_army("object-1", "hero", second),
                                      own_army("object-2", "town_garrison", [(6, 9)] + [(0, 0)] * 6)))
    # Through the wire, as a game delivers it: that is where unchanged parts become shared objects.
    return c.decode_observation(c.encode_observation(typed, schema=SCHEMA.history_schema), schema=SCHEMA.history_schema)


def layouts():
    empty = (0, 0)
    a = [(0, 20), (2, 7), empty, (4, 3), empty, empty, empty]
    b = [(6, 5), empty, empty, empty, empty, empty, (2, 1)]
    yield a, b, 1000
    yield [a[1], a[0], *a[2:]], b, 1000  # a swap inside the first army
    yield a, b, 1000  # and back: every part was seen before
    yield a, [b[0], (2, 4), *b[2:]], 1000  # the second army gains a stack
    yield a, [b[0], (2, 4), *b[2:]], 940  # only the gold changes
    yield [a[0], a[1], (0, 5), *a[3:]], [b[0], (2, 4), *b[2:]], 940


def frames():
    """A view and a decision per layout. No history checks them: only the encoder and the model read them."""
    made = []
    for revision, (first, second, gold) in enumerate(layouts()):
        seen = observation(first, second, gold)
        made.append(c.Frame("1", 0, len(made), 1, "view", observation=seen, schema=SCHEMA.history_schema))
        made.append(c.Frame("1", 0, len(made), 1, "decision", observation=seen, revision=revision,
                            offers=c.Offers.from_observation(seen, schema=SCHEMA.action_schema),
                            schema=SCHEMA.history_schema))
    return tuple(made)


def played(model, inputs, *, grad):
    """Every state, value, sampled action, probability and decision cache of a pass over the frames."""
    state, states, decisions = model.initial_state(), [], []
    generator = torch.Generator().manual_seed(5)
    with torch.set_grad_enabled(grad):
        for frame in inputs:
            output = model.forward_step(frame, state)
            state = output.state
            states.append(state.detach().clone())
            if output.decision is not None:
                distribution = output.decision.distribution
                action, evaluated = distribution.sample_evaluated(generator=generator)
                cache = {tree: tensor.detach().clone() for tree, tensor in distribution.cache.items() if type(tree) in (tuple, policy._Kept)}
                decisions.append((output.decision.value.detach().clone(), action, evaluated.log_prob.detach().clone(), cache))
    return states, decisions


def same(first, second):
    assert all(torch.equal(a, b) for a, b in zip(first[0], second[0], strict=True))
    for (value, action, log_prob, cache), (other_value, other_action, other_log_prob, other_cache) in zip(
            first[1], second[1], strict=True):
        assert torch.equal(value, other_value) and action == other_action and torch.equal(log_prob, other_log_prob)
        assert cache.keys() == other_cache.keys()
        assert all(torch.equal(tensor, other_cache[tree]) for tree, tensor in cache.items())


@pytest.fixture
def plain(monkeypatch):
    """Switch every reuse of this module's subject off: each value is decoded and each tree built anew."""
    def switch():
        monkeypatch.setattr(c, "_decoded_once", lambda decoder, raw, *context: decoder(raw, *context))
        monkeypatch.setattr(policy, "_KEPT_TYPES", frozenset())
        for kept in (c._DECODED, policy._KEPT_VALUES, policy._KEPT_PAIRS):
            kept.clear()
    return switch


def test_unchanged_parts_of_an_observation_are_the_same_objects():
    views = [frame.observation for frame in frames() if frame.kind == "view"]
    assert views[1].own_armies[0] is not views[0].own_armies[0] and views[1].own_armies[0] != views[0].own_armies[0]
    assert all(view.own_armies[2] is views[0].own_armies[2] and view.heroes[0] is views[0].heroes[0] for view in views)
    assert views[2].own_armies == views[0].own_armies and views[2].window.armies[0] is views[0].window.armies[0]
    # A stack that stayed in its slot is shared between two versions of its army.
    assert views[5].own_armies[0].slots[0] is views[0].own_armies[0].slots[0]
    trees = [policy.observation_semantics(view) for view in views]
    assert trees[2] == trees[0] and trees[1] != trees[0]
    assert type(policy._semantic(views[0].own_armies[2])) is policy._Kept
    assert policy._semantic(views[4].own_armies[2]) is policy._semantic(views[0].own_armies[2])


def test_a_kept_tree_is_the_tuple_it_was_built_as():
    tree = policy._node("army", policy._number(3), policy._node("role", policy._text("hero")))
    kept = policy._keep(tree)
    assert kept == tree and tree == kept and hash(kept) == hash(tree) and repr(kept) == repr(tree)
    holder, plain_holder = policy._node("own", kept, kept), policy._node("own", tree, tree)
    assert holder == plain_holder and hash(holder) == hash(plain_holder) and repr(holder) == repr(plain_holder)
    assert {plain_holder: 1}[holder] == 1 and {kept: 1}[tree] == 1
    assert kept.exact and policy._exact_tree(holder)
    signed = policy._keep(policy._node("army", ("number", -0.0)))
    assert not signed.exact and not policy._exact_tree(policy._node("own", signed))


def test_acting_with_reuse_computes_what_acting_without_it_computes(plain, monkeypatch):
    torch.manual_seed(3)
    model = ContextualActorCritic().eval()
    visits = []
    original = ContextualActorCritic._encode_visit
    monkeypatch.setattr(ContextualActorCritic, "_encode_visit",
                        lambda self, tree, cache, nodes=None: visits.append(tree) or original(self, tree, cache, nodes))
    reused = played(model, encode_history(frames(), guard=guard.VERSION), grad=False)
    first = len(visits)
    again = played(model, encode_history(frames(), guard=guard.VERSION), grad=False)  # everything was met before
    second = len(visits) - first
    assert model.__dict__["_memo_state"]["kept"]
    plain()
    untouched = ContextualActorCritic(model.config).eval()
    untouched.load_state_dict(model.state_dict())
    inputs = encode_history(frames(), guard=guard.VERSION)
    assert not any(type(tree) is policy._Kept for frame in inputs for tree in _subtrees(frame.semantics))
    del visits[:]
    reference = played(untouched, inputs, grad=True)  # no memo of any kind
    same(reused, reference)
    same(again, reference)
    del visits[:]
    same(played(untouched, inputs, grad=False), reference)  # the memo over plain trees, as before
    assert 2 * second < len(visits)  # trees walked per pass: far fewer once the unchanged parts are kept


def _subtrees(tree):
    stack = [tree]
    while stack:
        item = stack.pop()
        yield item
        if item[0] not in ("text", "number"):
            stack.extend(item[2])


def test_kept_encodings_never_outlive_the_weights(plain):
    torch.manual_seed(3)
    model = ContextualActorCritic().eval()
    inputs = encode_history(frames(), guard=guard.VERSION)
    before = played(model, inputs, grad=False)
    torch.manual_seed(4)
    other = ContextualActorCritic(model.config).eval()
    model.load_state_dict(other.state_dict())
    after = played(model, inputs, grad=False)
    assert not torch.equal(before[0][-1], after[0][-1])
    same(after, played(other, inputs, grad=True))


def test_a_game_compiles_into_the_same_recipe_with_and_without_reuse(plain):
    torch.manual_seed(3)
    model = ContextualActorCritic().eval()

    def recipe():
        inputs = encode_history(frames(), guard=guard.VERSION)
        actions = {}
        generator = torch.Generator().manual_seed(5)
        state = model.initial_state()
        with torch.no_grad():
            for frame in inputs:
                output = model.forward_step(frame, state)
                state = output.state
                if output.decision is not None:
                    actions[frame.frame.seq] = output.decision.distribution.sample_evaluated(generator=generator)[0]
        history = compile_history(model, inputs, actions)
        return actions, history.nodes, [(frame["root"], frame["conditionals"] is not None and [
            {name: value if not isinstance(value, torch.Tensor) else value.tolist() for name, value in item.items()}
            for item in frame["conditionals"]]) for frame in history.frames]

    reused = recipe()
    plain()
    assert recipe() == reused
