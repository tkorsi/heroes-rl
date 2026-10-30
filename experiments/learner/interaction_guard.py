"""Versioned restrictions on repeated, visible, free interaction.

This is an explicit policy-support intervention, not native legality or reward
shaping. Native offers and recorded history remain unchanged. A tracker consumes
only delivered player history; immutable snapshots can be rederived for PPO and
independent behavior reconstruction. Historical execution is disabled by default.
"""
from collections import Counter
from copy import copy
from dataclasses import dataclass

from . import contextual as c


VERSION = "visible-interaction-budget-v1"
REVEALED_VERSION = "visible-interaction-budget-revealed-v1"
RECRUIT_ONLY_VERSION = "visible-interaction-budget-recruit-only-v1"
SUPPORTED_VERSIONS = (VERSION, REVEALED_VERSION, RECRUIT_ONLY_VERSION)
# Temporary experiment restriction. Removing it requires a new explicit support
# selection; historical captures must continue to reconstruct this exact set.
# Whole-hero garrison changes are separate from moving stacks between slots.
RECRUIT_ONLY_BLOCKED_KINDS = frozenset({
    "split_stack", "merge_stacks", "swap_stacks", "transfer_stack",
    "open_stack", "select_upgrade", "upgrade_stack",
})
ACTIVE_VERSION = None
NAVIGATION_LIMIT = 2
MANAGEMENT_LIMIT = 32
NAVIGATION_KINDS = frozenset({
    "open_hero", "open_town", "open_panel", "open_building", "open_stack",
    "open_capability", "select_recruit", "select_hero", "select_upgrade",
    "inspect_object",
})
MANAGEMENT_KINDS = frozenset({
    "split_stack", "merge_stacks", "swap_stacks", "transfer_stack",
    "swap_garrison", "move_artifact", "assemble_artifact", "disassemble_artifact",
})
_DEFAULT = object()


def _mandatory(offers):
    return (offers.context.kind == "dialog" or
            offers.capability_panel is not None and offers.capability_panel.mandatory)


def nav_key(action, observation, offers):
    """Resolve optional navigation to displayed semantics, never revision refs.

    Recruitment controls have no semantic dwelling identifier in v8, so their
    fixed displayed position distinguishes controls. Slot identities deliberately
    exclude creature counts and layout, which free rearrangement can change.
    """
    if action.kind not in NAVIGATION_KINDS or _mandatory(offers):
        return None
    prefix = (action.kind, action.subject or "")
    if action.kind in ("open_hero", "open_town"):
        return prefix
    if action.kind == "open_capability":
        return (*prefix, action.panel)
    if action.kind == "open_panel":
        controls = offers.panel.controls
        if action.panel != "recruitment":
            return (*prefix, action.panel)
        entries = [item for item in controls if item.panel == "recruitment"]
        index = next(index for index, item in enumerate(entries) if item.ref == action.selection)
        return (*prefix, action.panel, str(index))
    if action.kind == "open_building":
        item = next(item for item in offers.panel.buildings if item.ref == action.selection)
        return (*prefix, item.name_text_id)
    if action.kind == "select_recruit":
        item = next(item for item in offers.panel.recruitment.offers if item.ref == action.selection)
        return (*prefix, str(item.creature))
    if action.kind == "select_hero":
        item = next(item for item in offers.panel.tavern.offers if item.ref == action.selection)
        return (*prefix, item.name_text_id, str(item.portrait))
    if action.kind == "open_stack":
        army, slot = c.army_slot(offers.armies, action.selection)
        return (action.kind, army.subject, str(slot.slot))
    if action.kind == "select_upgrade":
        item = next(item for item in offers.stack_panel.offers if item.ref == action.selection)
        return (*prefix, str(offers.stack_panel.source.slot), str(item.creature))
    item = next(item for item in offers.capability_panel.choices if item.ref == action.selection)
    return (*prefix, item.destination)


def management_subjects(action, observation, offers):
    """Charge both actors, preventing budget renewal by reversing a transfer."""
    if action.kind not in MANAGEMENT_KINDS:
        return ()
    subjects = set()
    if action.kind in ("move_artifact", "assemble_artifact", "disassemble_artifact"):
        choice = next(item for item in offers.capability_panel.choices if item.ref == action.selection)
        subjects.add(choice.subject)
        if choice.destination is not None:
            subjects.add(choice.destination)
    elif action.kind == "swap_garrison":
        subjects.add(action.subject)
        subjects.update(army.subject for army in offers.armies)
    else:
        armies = offers.armies or ()
        if action.kind == "transfer_stack" and offers.panel is not None and offers.panel.armies is not None:
            exchange = offers.panel.armies
            armies = (exchange.source,) + ((exchange.destination,) if exchange.destination else ())
        for ref in (action.selection, action.destination):
            army, _ = c.army_slot(armies, ref)
            if army is not None:
                subjects.add(army.subject)
    return tuple(sorted(subjects or {action.subject}))


def _strings(values, name):
    if type(values) is not tuple or any(type(item) is not str or not item for item in values):
        raise ValueError(f"Expected immutable nonempty {name} strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"Expected unique sorted {name}")


@dataclass(frozen=True)
class GuardSupport:
    version: str
    blocked_navigation: tuple[tuple[str, ...], ...] = ()
    blocked_management_subjects: tuple[str, ...] = ()
    movable_heroes: tuple[str, ...] = ()
    blocked_coordinates: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError("Unsupported interaction guard version")
        _strings(self.blocked_management_subjects, "management subjects")
        _strings(self.movable_heroes, "movable heroes")
        if type(self.blocked_navigation) is not tuple:
            raise ValueError("Expected immutable navigation keys")
        for key in self.blocked_navigation:
            if (type(key) is not tuple or len(key) < 2 or key[0] not in NAVIGATION_KINDS
                    or any(type(item) is not str or not item for item in key)):
                raise ValueError("Malformed semantic navigation key")
        if self.blocked_navigation != tuple(sorted(set(self.blocked_navigation))):
            raise ValueError("Expected unique sorted navigation keys")
        if type(self.blocked_coordinates) is not tuple:
            raise ValueError("Expected immutable coordinates")
        for item in self.blocked_coordinates:
            if (type(item) is not tuple or len(item) != 3 or type(item[0]) is not str or not item[0]
                    or any(type(value) is not int or value < 0 for value in item[1:])):
                raise ValueError("Malformed blocked movement coordinate")
        if self.blocked_coordinates != tuple(sorted(set(self.blocked_coordinates))):
            raise ValueError("Expected unique sorted blocked coordinates")

    def permits(self, action, observation, offers):
        if not offers.permits(action):
            return False
        if self.version == RECRUIT_ONLY_VERSION and action.kind in RECRUIT_ONLY_BLOCKED_KINDS:
            return False
        if _mandatory(offers) or action.kind in ("close", "back", "close_capability", "end_turn"):
            return True
        if action.kind == "move_to":
            return self._permits_movement(action.subject, action.x, action.y,
                                          self._revealed_coordinates(action.subject, observation))
        if action.kind in MANAGEMENT_KINDS:
            return not set(management_subjects(action, observation, offers)).intersection(self.blocked_management_subjects)
        return nav_key(action, observation, offers) not in self.blocked_navigation

    def _revealed_coordinates(self, subject, observation):
        if self.version == VERSION:
            return None
        # Membership in the current delivered observation is the entire added
        # rule. Blocked/visitable flags and route feasibility are not consulted.
        level = next((hero.position[2] for hero in observation.heroes if hero.handle == subject), None)
        return frozenset((tile.position[0], tile.position[1]) for tile in observation.tiles
                         if tile.position[2] == level)

    def _permits_movement(self, subject, x, y, revealed):
        return (subject in self.movable_heroes
                and (subject, x, y) not in self.blocked_coordinates
                and (revealed is None or (x, y) in revealed))

    def movement_coordinates(self, subject, observation, offers):
        """Current permitted coordinates, in the historical row-major order."""
        width, height, _ = offers.size
        revealed = self._revealed_coordinates(subject, observation)
        indices = (range(width * height) if revealed is None else
                   sorted(y * width + x for x, y in revealed))
        return (index for index in indices
                if offers.permits(c.Action("move_to", subject, index % width, index // width,
                                           schema=offers.schema))
                and self._permits_movement(subject, index % width, index // width, revealed))

    def to_dict(self):
        return {
            "version": self.version,
            "blocked_navigation": [list(key) for key in self.blocked_navigation],
            "blocked_management_subjects": list(self.blocked_management_subjects),
            "movable_heroes": list(self.movable_heroes),
            "blocked_coordinates": [list(item) for item in self.blocked_coordinates],
        }

    @classmethod
    def from_dict(cls, value):
        names = {"version", "blocked_navigation", "blocked_management_subjects", "movable_heroes", "blocked_coordinates"}
        if type(value) is not dict or set(value) != names:
            raise ValueError("Malformed interaction guard support")
        if any(type(value[name]) is not list for name in names - {"version"}):
            raise ValueError("Guard support collections must be JSON arrays")
        if any(type(item) is not list for name in ("blocked_navigation", "blocked_coordinates") for item in value[name]):
            raise ValueError("Guard support rows must be JSON arrays")
        return cls(value["version"], tuple(tuple(key) for key in value["blocked_navigation"]),
                   tuple(value["blocked_management_subjects"]), tuple(value["movable_heroes"]),
                   tuple(tuple(item) for item in value["blocked_coordinates"]))


def _world(observation):
    # No window, army layout, artifact inventory, query reference or revision.
    # Garrison swaps and artifact-derived movement maxima do not renew budgets.
    return (observation.size, tuple(sorted(observation.tiles, key=lambda item: item.position)),
            tuple(sorted(observation.objects, key=lambda item: item.handle)), observation.resources,
            tuple(sorted((h.handle, h.position, h.movement) for h in observation.heroes)),
            tuple(sorted(observation.towns, key=lambda item: item.handle)))


class Tracker:
    """A per-player, per-episode chronological history reducer, not a model."""

    def __init__(self, version=_DEFAULT):
        self.version = ACTIVE_VERSION if version is _DEFAULT else version
        if self.version is not None and self.version not in SUPPORTED_VERSIONS:
            raise ValueError("Unsupported interaction guard version")
        self.day = None
        self.identity = None
        self.last_seq = -1
        self.world = None
        self.navigation = Counter()
        self.management = Counter()
        self.failed_coordinates = set()
        self.decision = None
        self.action = None
        self.consumed = False
        self.epoch = 0
        self.action_epoch = None

    def copy(self):
        # Frames/observations are immutable and may contain large visible maps.
        # Transactional encoders copy only the three mutable accumulators.
        result = copy(self)
        result.navigation = self.navigation.copy()
        result.management = self.management.copy()
        result.failed_coordinates = self.failed_coordinates.copy()
        return result

    def _progress(self):
        self.navigation.clear()
        self.failed_coordinates.clear()
        self.epoch += 1

    def observe(self, frame):
        if self.version is None:
            return None
        if type(frame) is not c.Frame:
            raise ValueError("Interaction guard requires a typed visible frame")
        identity = (frame.episode, frame.player)
        if self.identity not in (None, identity) or frame.seq <= self.last_seq:
            raise ValueError("Interaction guard requires one ordered player episode")
        self.identity, self.last_seq = identity, frame.seq
        if self.day != frame.day:
            if self.day is not None and frame.day < self.day:
                raise ValueError("Interaction guard day cannot move backwards")
            self.day = frame.day
            self.management.clear()
            self._progress()
        if frame.observation is not None:
            world = _world(frame.observation)
            # A free management operation may alter visible garrison placement or
            # artifact-derived stats; it must not renew the navigation budget.
            management = self.consumed and self.action is not None and self.action.kind in MANAGEMENT_KINDS
            if self.world is not None and world != self.world and not management:
                self._progress()
            self.world = world
        if frame.kind == "decision":
            self.decision, self.action, self.consumed = frame, None, False
            movable = tuple(sorted(h.handle for h in frame.observation.heroes if h.movement > 0 and not h.in_garrison))
            blocked = self.failed_coordinates | {(h.handle, h.position[0], h.position[1]) for h in frame.observation.heroes}
            return GuardSupport(self.version,
                tuple(sorted(key for key, count in self.navigation.items() if count >= NAVIGATION_LIMIT)),
                tuple(sorted(subject for subject, count in self.management.items() if count >= MANAGEMENT_LIMIT)),
                movable, tuple(sorted(blocked)))
        if frame.kind == "submitted":
            self.action = frame.action
        elif frame.kind == "consumed":
            if self.decision is None or self.action is None or self.consumed:
                raise ValueError("Interaction guard requires one submitted action per consumption")
            self.consumed = True
            self.action_epoch = self.epoch
            obs, offers = self.decision.observation, self.decision.offers
            key = nav_key(self.action, obs, offers)
            if key is not None:
                self.navigation[key] += 1
            if not _mandatory(offers):
                for subject in management_subjects(self.action, obs, offers):
                    self.management[subject] += 1
        elif frame.kind == "event" and frame.event.kind in ("battle_start", "battle_complete", "hero_lost", "level_gained"):
            self._progress()
        elif frame.kind == "result" and self.action is not None and self.consumed:
            result = frame.result
            if (result.accepted and (result.status in ("built", "recruited", "hired", "upgraded")
                    or result.status == "capability_applied" and self.action.kind in
                        ("trade_resource", "buy_artifact", "choose_level_up"))):
                self._progress()
            if self.action.kind == "move_to" and result.status in ("no_route_known", "movement_limit", "hero_unavailable", "reached", "rejected"):
                hero = next((h for h in self.decision.observation.heroes if h.handle == self.action.subject), None)
                unchanged = (hero is not None and result.position in (None, hero.position)
                             and result.movement in (None, hero.movement))
                if unchanged and self.action_epoch == self.epoch:
                    self.failed_coordinates.add((self.action.subject, self.action.x, self.action.y))
        return None
