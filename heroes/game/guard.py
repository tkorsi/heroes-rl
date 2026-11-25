"""The interaction guard: actions the policy may not sample although the game offers them.

Self-play would otherwise spend its time on free actions that change nothing:
reopening the same window, shuffling stacks back and forth, walking into the
same wall. The guard removes those from the policy's choices. It is a
restriction on what is sampled, not game legality and not reward shaping: the
engine's offers and the recorded history stay as they are.

A ``Tracker`` follows one seat's frames and, at every decision, returns the
``GuardSupport`` that says what is currently excluded:

* a navigation step already taken twice since the visible world last changed
* any stack management by a hero or town that already did 32 such moves today
* a swap that would recreate an army layout already seen since the last progress
* a move to an unrevealed tile, to the hero's own tile, to its previous target
  or to a target that failed since the last progress
* repeating the previous command exactly, and reopening a town right after closing it

Three daily budgets can be set through the guard's name (``version``). They
change what a policy may sample and so the games it plays; the standard guard
has only the first, at 32:

* ``stack-moves``: stack management per hero or town
* ``windows``: optional window navigation (opening a hero, a town, a panel, a
  stack, a capability) per hero or town
* ``decisions``: answers of the whole seat; past it the seat may only close
  what is open, answer what it must and end its turn
"""
from collections import Counter
from dataclasses import dataclass

from . import contract as c

VERSION = "visible-interaction-budget-hero-relative-stack-cycle-v1"
NAVIGATION_LIMIT = 2
MANAGEMENT_LIMIT = 32
_BUDGETS = ("stack-moves", "windows", "decisions")


def version(*, stack_moves=None, windows=None, decisions=None):
    """The guard's name with these daily budgets: ``VERSION`` itself for the standard ones."""
    name = VERSION
    for label, value in zip(_BUDGETS, (stack_moves, windows, decisions)):
        if value is not None and (label, value) != ("stack-moves", MANAGEMENT_LIMIT):
            if type(value) is not int or value < 1:
                raise ValueError("A daily budget is a positive integer")
            name += f"+{label}-{value}"
    return name


def budgets(name):
    """``(stack moves, windows, decisions)`` of a guard name; None where the guard sets no budget."""
    base, *parts = name.split("+")
    values = dict.fromkeys(_BUDGETS)
    for part in parts:
        label, _, value = part.rpartition("-")
        values[label] = int(value) if value.isdigit() else None
    found = (values["stack-moves"] or MANAGEMENT_LIMIT, values["windows"], values["decisions"])
    # Only the one spelling ``version`` gives is a name: the same guard never has two.
    if base != VERSION or 0 in found or version(stack_moves=found[0], windows=found[1], decisions=found[2]) != name:
        raise ValueError(f"Unsupported interaction guard {name!r}; this code implements {VERSION}")
    return found


NAVIGATION_KINDS = frozenset({
    "open_hero", "open_town", "open_panel", "open_building", "open_stack",
    "open_capability", "select_recruit", "select_hero", "select_upgrade",
    "inspect_object",
})
MANAGEMENT_KINDS = frozenset({
    "split_stack", "merge_stacks", "swap_stacks", "transfer_stack",
    "swap_garrison", "move_artifact", "assemble_artifact", "disassemble_artifact",
})


_ARRANGEMENTS = frozenset(("split_stack", "merge_stacks", "swap_stacks"))  # charged to the armies of offers.armies


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


def _command_key(action, observation, offers):
    """Identify a command by visible targets, without revision-scoped refs."""
    kind, context = action.kind, offers.context
    navigation = nav_key(action, observation, offers)
    if navigation is not None:
        target = navigation[2:]
    elif kind == "move_to":
        target = (action.x, action.y)
    elif kind == "recruit":
        panel = offers.panel.recruitment
        card = next(card for card in panel.offers if card.ref == action.selection)
        target = (card.creature, panel.destination.role)
    elif kind == "construct":
        target = (offers.panel.building.name_text_id,)
    elif kind == "hire_hero":
        card = next(card for card in offers.panel.tavern.offers if card.ref == action.selection)
        target = (card.name_text_id, card.portrait)
    elif kind in c.CAPABILITY_ACTIONS:
        choice = next(choice for choice in offers.capability_panel.choices if choice.ref == action.selection)
        target = (choice.artifact, choice.slot, choice.destination, choice.destination_slot,
                  choice.spell, choice.skill, choice.skill_level, choice.resource_from,
                  choice.resource_to, choice.x, choice.y)
    else:
        target = (action.selection, action.destination)
    return tuple("" if item is None else str(item) for item in (
        kind, context.kind, context.panel, context.subject, action.subject, *target, action.count))


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


def _swap_key(armies, source, destination):
    left, a = c.army_slot(armies, source)
    right, b = c.army_slot(armies, destination)
    first, second = sorted(((left.subject, a.slot), (right.subject, b.slot)))
    return (*first, *second)


def _army_layout(observation, armies):
    # Semantic subjects/slots survive window closure and revision-ref changes.
    rows = {}
    for army in observation.own_armies or ():
        slots = [(None, 0)] * 7
        for slot in army.slots:
            slots[slot.slot] = (slot.creature, slot.count)
        rows[army.subject] = tuple(slots)
    for army in armies:
        rows[army.subject] = tuple((slot.creature, slot.count) for slot in army.slots)
    return tuple(sorted(rows.items()))


def _blocked_swaps(observation, offers, seen):
    armies = offers.armies or ()
    if not armies:
        return ()
    layout = _army_layout(observation, armies)
    return tuple(sorted({key for key, swapped in _swapped_layouts(layout, armies) if swapped in seen}))


_SWAP_KEYS = {}  # ids of the armies -> (the armies, the keys of the swaps they offer, each once, in order)


def _swap_keys(armies):
    """The key of every swap ``armies`` offer, each once, in the order the offers list them: a function of the armies."""
    cache_key = tuple(map(id, armies))
    known = _SWAP_KEYS.get(cache_key)
    if known is not None and all(held is army for held, army in zip(known[0], armies)):
        return known[1]
    keys = tuple(dict.fromkeys(_swap_key(armies, source, destination)
                               for source, destination, _ in c.army_operation_choices(armies, "swap_stacks")))
    if len(_SWAP_KEYS) >= 256:
        _SWAP_KEYS.clear()  # bounded memory; a miss only computes again
    _SWAP_KEYS[cache_key] = (armies, keys)  # holds the armies, so their ids name nothing else
    return keys


_SWAPPED = {}  # (layout, ids of the armies) -> (the armies, ((swap key, the layout after that swap), ...))


def _swapped_layouts(layout, armies):
    """Each swap the armies offer, by its key, with the layout it would leave: a function of the two alone.

    A seat decides again and again in the same window; only the layouts it has seen grow.
    """
    cache_key = (layout, *map(id, armies))
    known = _SWAPPED.get(cache_key)
    if known is not None and all(held is army for held, army in zip(known[0], armies)):
        return known[1]
    found = {}
    # The layout is sorted by subject (each subject once) and a swap keeps the subjects: the swapped layout,
    # sorted, has its rows in the same places.
    places = {subject: index for index, (subject, _) in enumerate(layout)}
    for key in _swap_keys(armies):
        left, a, right, b = key
        i, j = places[left], places[right]
        x, y = layout[i][1][a], layout[j][1][b]
        rows = list(layout)
        slots = list(rows[i][1]); slots[a] = y; rows[i] = (left, tuple(slots))
        slots = list(rows[j][1]); slots[b] = x; rows[j] = (right, tuple(slots))
        found[key] = tuple(rows)
    if len(_SWAPPED) >= 256:
        _SWAPPED.clear()  # bounded memory; a miss only computes again
    _SWAPPED[cache_key] = (armies, tuple(found.items()))  # holds the armies, so their ids name nothing else
    return _SWAPPED[cache_key][1]


@dataclass(frozen=True)
class GuardSupport:
    blocked_navigation: tuple[tuple[str, ...], ...] = ()
    blocked_management_subjects: tuple[str, ...] = ()
    movable_heroes: tuple[str, ...] = ()
    blocked_coordinates: tuple[tuple[str, int, int], ...] = ()
    previous_command: tuple[str, ...] | None = None
    blocked_swaps: tuple[tuple[str, int, str, int], ...] = ()
    blocked_window_subjects: tuple[str, ...] = ()  # heroes and towns past their daily window budget
    exhausted: bool = False  # the seat is past its daily decision budget

    def permits(self, action, observation, offers):
        kind = action.kind if type(action) is c.Action else None
        if (kind in _ARRANGEMENTS and self.blocked_management_subjects and offers.armies
                and not _mandatory(offers) and kind not in ("close", "back", "close_capability")):
            # Every path below ends in False for a rearrangement whose subject is past its budget, whether or not
            # the offers permit it: answered here without the offers' and the repeat's checks.
            army_slot = c.army_slot
            subjects = {army.subject for army, _ in (army_slot(offers.armies, action.selection),
                                                     army_slot(offers.armies, action.destination)) if army is not None}
            if (subjects or {action.subject}) & set(self.blocked_management_subjects):
                return False
        swap_key = None
        if kind == "swap_stacks" and self.blocked_swaps and offers.armies and not _mandatory(offers):
            # So does a swap the layouts already seen exclude.
            left, a = c.army_slot(offers.armies, action.selection)
            right, b = c.army_slot(offers.armies, action.destination)
            if a is not None and b is not None:
                swap_key = _swap_key(offers.armies, action.selection, action.destination)
                if swap_key in self.blocked_swaps:
                    return False
        if not offers.permits(action):
            return False
        if _mandatory(offers) or action.kind in ("close", "back", "close_capability"):
            return True
        if self.exhausted:
            return action.kind == "end_turn"
        if self._repeats_command(action, observation, offers):
            return action.kind == "end_turn" and not self._map_has_alternative(observation, offers)
        if action.kind == "end_turn":
            return True
        if action.kind == "move_to":
            return self._permits_movement(action.subject, action.x, action.y,
                                          self._revealed_coordinates(action.subject, observation))
        if action.kind == "swap_stacks":
            if self.blocked_swaps and (swap_key if swap_key is not None else
                                       _swap_key(offers.armies, action.selection, action.destination)) in self.blocked_swaps:
                return False
        if action.kind in MANAGEMENT_KINDS:
            # Nothing is blocked: no subject can be. Otherwise the subjects charged, as management_subjects lists them.
            return not self.blocked_management_subjects or not set(
                management_subjects(action, observation, offers)).intersection(self.blocked_management_subjects)
        key = nav_key(action, observation, offers)
        return key not in self.blocked_navigation and (key is None or key[1] not in self.blocked_window_subjects)

    def _repeats_command(self, action, observation, offers):
        return (self.previous_command is not None and self.previous_command[0] == action.kind
                and self.previous_command == _command_key(action, observation, offers))

    def _map_has_alternative(self, observation, offers):
        return (any(self.permits(c.Action(kind, subject, schema=offers.schema), observation, offers)
                    for kind, refs in (("open_hero", offers.hero_refs), ("open_town", offers.town_refs))
                    for subject in refs)
                or any(next(self.movement_coordinates(subject, observation, offers), None) is not None
                       for subject in self.movable_heroes))

    def repeated_quantity(self, action, observation, offers):
        previous = self.previous_command
        if (previous is not None and previous[0] == action.kind and not _mandatory(offers)
                and previous[:-1] == _command_key(action, observation, offers)[:-1]):
            return int(previous[-1])
        return None

    def _revealed_coordinates(self, subject, observation):
        # Membership in the current observation is the whole rule. Blocked and
        # visitable flags and route feasibility are not consulted.
        level = next((hero.position[2] for hero in observation.heroes if hero.handle == subject), None)
        return _revealed(observation.tiles, level)

    def _permits_movement(self, subject, x, y, revealed):
        return (subject in self.movable_heroes
                and (subject, x, y) not in self.blocked_coordinates
                and (x, y) in revealed)

    def movement_coordinates(self, subject, observation, offers):
        """The coordinates the hero may move to, as row-major indices in ascending order."""
        width, height, _ = offers.size
        revealed = self._revealed_coordinates(subject, observation)
        return (index for index in sorted(y * width + x for x, y in revealed)
                if offers.permits(c.Action("move_to", subject, index % width, index // width,
                                           schema=offers.schema))
                and self._permits_movement(subject, index % width, index // width, revealed))


_REVEALED = {}  # (id of a tile tuple, level) -> (the tuple, the cells of that level it shows)


def _revealed(tiles, level):
    """The ``(x, y)`` of every tile of ``level`` in ``tiles``. Most observations carry the tile tuple of the one before."""
    known = _REVEALED.get((id(tiles), level))
    if known is None or known[0] is not tiles:
        known = (tiles, frozenset((tile.position[0], tile.position[1]) for tile in tiles if tile.position[2] == level))
        if len(_REVEALED) >= 8:
            _REVEALED.clear()  # the entry holds its tuple, so an id names nothing else while it is kept
        _REVEALED[id(tiles), level] = known
    return known[1]


_SORTED_TILES = [None, None]  # the tile tuple sorted last, and its tiles by position


def _tiles_by_position(tiles):
    # Most observations carry the very tile tuple of the one before.
    if tiles is not _SORTED_TILES[0]:
        _SORTED_TILES[:] = tiles, tuple(sorted(tiles, key=lambda item: item.position))
    return _SORTED_TILES[1]


_WORLDS = {}  # id of an observation -> (the observation, its world)


def _world(observation):
    # A repeated view is the same observation object (protocol): its world is built once.
    known = _WORLDS.get(id(observation))
    if known is not None and known[0] is observation:
        return known[1]
    world = _build_world(observation)
    if len(_WORLDS) >= 2048:
        _WORLDS.clear()  # bounded memory; a miss only builds again
    _WORLDS[id(observation)] = (observation, world)  # holds the observation, so its id names nothing else
    return world


def _build_world(observation):
    # No window, army layout, artifact inventory, query reference or revision.
    # Garrison swaps and artifact-derived movement maxima do not renew budgets.
    return (observation.size, _tiles_by_position(observation.tiles),
            tuple(sorted(observation.objects, key=lambda item: item.handle)), observation.resources,
            tuple(sorted((h.handle, h.position, h.movement) for h in observation.heroes)),
            tuple(sorted(observation.towns, key=lambda item: item.handle)))


class Tracker:
    """Follows one seat's frames in order. ``Tracker(None)`` guards nothing."""

    def __init__(self, version=None):
        self.management_limit, self.window_limit, self.decision_limit = (
            (MANAGEMENT_LIMIT, None, None) if version is None else budgets(version))
        self.version = version
        self.windows = Counter()  # optional navigation today, by hero or town
        self.decisions = 0  # answers the engine consumed today
        self.day = None
        self.identity = None
        self.last_seq = -1
        self.world = None
        self.navigation = Counter()
        self.management = Counter()
        self.failed_coordinates = set()
        self.army_layouts = set()
        self.town_reopen_blocked = False
        self.previous_command = None
        self.previous_move = None
        self.decision = None
        self.action = None
        self.consumed = False
        self.epoch = 0
        self.action_epoch = None

    def _progress(self):
        self.navigation.clear()
        self.failed_coordinates.clear()
        self.army_layouts.clear()
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
            self.windows.clear()
            self.decisions = 0
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
            if self.previous_move is not None:
                blocked.add(self.previous_move)
            navigation = {key for key, count in self.navigation.items() if count >= NAVIGATION_LIMIT}
            if self.town_reopen_blocked:
                navigation.update(("open_town", subject) for subject in frame.offers.town_refs)
            self.army_layouts.add(_army_layout(frame.observation, frame.offers.armies or ()))
            blocked_swaps = _blocked_swaps(frame.observation, frame.offers, self.army_layouts)
            return GuardSupport(
                tuple(sorted(navigation)),
                tuple(sorted(subject for subject, count in self.management.items() if count >= self.management_limit)),
                movable, tuple(sorted(blocked)), self.previous_command, blocked_swaps,
                () if self.window_limit is None else
                tuple(sorted(subject for subject, count in self.windows.items() if count >= self.window_limit)),
                self.decision_limit is not None and self.decisions >= self.decision_limit)
        if frame.kind == "submitted":
            self.action = frame.action
        elif frame.kind == "consumed":
            if self.decision is None or self.action is None or self.consumed:
                raise ValueError("Interaction guard requires one submitted action per consumption")
            self.consumed = True
            self.action_epoch = self.epoch
            obs, offers = self.decision.observation, self.decision.offers
            self.previous_command = _command_key(self.action, obs, offers)
            self.previous_move = ((self.action.subject, self.action.x, self.action.y)
                                  if self.action.kind == "move_to" else None)
            key = nav_key(self.action, obs, offers)
            if key is not None:
                self.navigation[key] += 1
                self.windows[key[1]] += 1
            self.decisions += not _mandatory(offers)
            if not _mandatory(offers):
                for subject in management_subjects(self.action, obs, offers):
                    self.management[subject] += 1
        elif frame.kind == "event" and frame.event.kind in ("battle_start", "battle_complete", "hero_lost", "level_gained"):
            self._progress()
        elif frame.kind == "result" and self.action is not None and self.consumed:
            result = frame.result
            if result.accepted:
                # Only a resolved player choice advances this cooldown. Changes
                # in the visible world may reset the separate navigation budget.
                self.town_reopen_blocked = (self.action.kind == "close" and result.status == "closed"
                                            and self.decision.offers.context.kind == "town")
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
