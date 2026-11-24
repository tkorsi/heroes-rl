"""Recognize successive empty visits to the same optional window in one seat's history."""
from collections import deque
from itertools import chain
from typing import NamedTuple

from . import contract, guard


_OPEN = {"open_hero", "open_town", "open_panel", "open_building", "open_stack", "open_capability"}
_CLOSE = {"close", "back", "close_capability"}


def _same_window(left, right):
    if left is right or left == right:
        return True
    # On reopening, control IDs can change. Compare all displayed fields and
    # require every use of a control ID to refer to the same corresponding control.
    forward, reverse = {}, {}

    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, tuple):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if not hasattr(a, "__dataclass_fields__"):
            return a == b
        for name in contract._field_names(type(a)):
            x, y = getattr(a, name), getattr(b, name)
            binding = (name == "ref" or name.endswith("_ref")
                       or name == "selection" and isinstance(a, contract.Context) and a.kind != "exchange")
            if binding and isinstance(x, str) and isinstance(y, str):
                if forward.setdefault(x, y) != y or reverse.setdefault(y, x) != x:
                    return False
            elif not equal(x, y):
                return False
        return True

    return equal(left, right)


def _same_members(left, right):
    # A map refresh can reorder its uniquely identified objects. Most adjacent
    # frames share the original immutable tuples and need no hashing at all.
    return left == right or (left is not None and right is not None and set(left) == set(right))


def _same_world(a, b):
    return a is b or (a.size == b.size and a.resources == b.resources and a.heroes == b.heroes
                      and a.towns == b.towns and a.army_information == b.army_information
                      and _same_members(a.own_armies, b.own_armies) and _same_members(a.tiles, b.tiles)
                      and _same_members(a.objects, b.objects))


def _same_view(left, right):
    a, b = left.observation, right.observation
    return left.day == right.day and _same_world(a, b) and _same_window(a.window, b.window)


class _Step(NamedTuple):
    before: contract.Frame
    after: contract.Frame
    target: tuple | None  # semantic target for an open, None for a close


def _navigation_step(frames, start, stop):
    before = frames[start]
    action = after = None
    context = before.observation.context
    for i in range(start + 1, stop):
        frame = frames[i]
        if frame.kind == "submitted":
            if frame.decision_seq != before.seq:
                return None
            action = frame.action
            if action.kind not in _OPEN and action.kind not in _CLOSE:
                return None
        elif frame.kind == "context":
            context = frame.context
        elif frame.kind == "view":
            after = frame
        elif frame.kind == "result":
            if action is None or frame.decision_seq != before.seq or not frame.result.accepted:
                return None
            if action.kind in _OPEN and frame.result.status == "opened":
                target = guard.nav_key(action, before.observation, before.offers)
                if target is None:  # mandatory navigation
                    return None
            elif action.kind in _CLOSE and frame.result.status == "closed":
                target = None
            else:
                return None
            # Engines can deliver the updated view on either side of the result.
            # An unanswered next decision can supply it as well.
            if after is None or after.observation.context != context:
                after = next((frames[j] for j in range(i + 1, min(stop + 1, len(frames)))
                              if frames[j].kind in ("view", "decision")), None)
            if after is not None and after.observation.context == context:
                return _Step(before, after, target)
            return None
    return None


class WindowMemory:
    """Previously inspected targets visible to one seat, as of each decision.

    Record a voluntary open only once an accepted, immediate close returns to
    the original visible state. Keep independent targets while the seat browses.
    """

    def __init__(self):
        self.day = None
        self.observation = None
        self.parents = {}  # parent context -> (last visible window there, immutable target keys)
        self.frames = []  # one decision through its successor
        self.opened = None  # confirmed open immediately preceding the current decision

    def _at(self, observation):
        window = observation.window
        parent = None if window is None else (window.context.kind, window.context.subject, window.context.panel)
        known = self.parents.get(parent)
        if known is not None:
            if _same_window(known[0], window):
                return parent, known[1]
            del self.parents[parent]
        return parent, frozenset()

    def observe(self, frame):
        """Return the stale optional navigation keys for this decision, if any."""
        observation = frame.observation
        changed_world = (observation is not None and self.observation is not None
                         and not _same_world(self.observation, observation))
        # Some changes (e.g. new stock, a building or equipment) are known from
        # an accepted command even before the affected window is visible again.
        changed_by_action = (frame.result is not None and frame.result.accepted and frame.result.status in
                             {"built", "recruited", "hired", "upgraded", "transferred", "split", "merged", "swapped",
                              "garrison_swapped", "capability_applied", "dialog_answered", "interaction"})
        if frame.day != self.day or changed_world or changed_by_action or frame.kind == "event":
            self.parents.clear()
            self.frames.clear()
            self.opened = None
        self.day = frame.day
        if observation is not None:
            self.observation = observation

        if self.frames:
            self.frames.append(frame)
        if frame.kind != "decision":
            return frozenset()
        step = _navigation_step(self.frames, 0, len(self.frames) - 1) if self.frames else None
        self.frames = [frame]
        previous = self.opened
        self.opened = step if step is not None and step.target is not None else None
        if self.opened is not None:
            # Re-entering begins a new visit whose contents and result may differ.
            source = step.before.observation
            parent, keys = self._at(source)
            if step.target in keys:
                self.parents[parent] = (source.window, keys - {step.target})
        elif step is not None and previous is not None and _same_view(previous.after, step.before):
            if _same_view(previous.before, step.after) and _same_view(step.after, frame):
                parent, keys = self._at(observation)
                self.parents[parent] = (observation.window, keys | {previous.target})
        return self._at(observation)[1]


def repeated_empty_opens(frames):
    """Return the second OPEN's decision seq for each confirmed OPEN/CLOSE/OPEN/CLOSE.

    Each visit must return to its original parent view and both visits must
    show identical content. Adjacent pairs overlap when a bot keeps cycling.
    """
    decisions = (frame.seq for frame in frames if frame.kind == "decision")
    start = next(decisions, None)
    if start is None:
        return ()
    recent = deque(maxlen=4)
    repeats = []
    for stop in chain(decisions, (len(frames),)):
        step = _navigation_step(frames, start, stop)
        start = stop
        if step is None:
            recent.clear()
            continue
        recent.append(step)
        if len(recent) == 4:
            a, b, c, d = recent
            if (a.target is not None and a.target == c.target and b.target is None and d.target is None
                    and a.before.day == d.after.day
                    and all(_same_view(a.before, view) for view in (b.after, c.before, d.after))
                    and all(_same_view(a.after, view) for view in (b.before, c.after, d.before))):
                repeats.append(c.before.seq)
    return tuple(repeats)


class RepeatedOpens:
    """``repeated_empty_opens`` for a history that is still growing.

    ``read`` takes the seat's frames so far and settles every decision whose
    successor has arrived; ``finish`` settles the last one and returns what
    ``repeated_empty_opens`` returns for the finished history. A frame before
    ``needed`` is not read again, so its observation may be dropped.
    """

    def __init__(self):
        self.recent = deque(maxlen=4)
        self.repeats = []
        self.start = None  # the latest decision; its step is settled when the next decision is there
        self.scanned = 0  # frames looked at for decisions

    @property
    def needed(self):
        """The first frame a later call still reads."""
        return self.scanned if self.start is None else self.start

    def read(self, frames):
        for position in range(self.scanned, len(frames)):
            if frames[position].kind == "decision":
                if self.start is not None:
                    self._settle(frames, self.start, position)
                self.start = position
        self.scanned = len(frames)

    def finish(self, frames):
        self.read(frames)
        if self.start is not None:
            self._settle(frames, self.start, len(frames))
            self.start = None
        return tuple(self.repeats)

    def _settle(self, frames, start, stop):
        step = _navigation_step(frames, start, stop)
        recent = self.recent
        if step is None:
            recent.clear()
            return
        recent.append(step)
        if len(recent) == 4:
            a, b, c, d = recent
            if (a.target is not None and a.target == c.target and b.target is None and d.target is None
                    and a.before.day == d.after.day
                    and all(_same_view(a.before, view) for view in (b.after, c.before, d.after))
                    and all(_same_view(a.after, view) for view in (b.before, c.after, d.before))):
                self.repeats.append(c.before.seq)
