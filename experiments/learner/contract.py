"""Strict player-history v1 boundary. See README for producer obligations.

Validation establishes shape and ordering, not the truth of visibility claims.
Only an audited environment adapter may produce these records from the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SCHEMA = "heroes-player-history-v1"
ACTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "end_turn")
BOARD_SIZE = (20, 20, 1)
TERRAIN_TYPES = 16
OBJECT_TYPES = 256
OBJECT_SUBTYPES = 256
CREATURE_TYPES = 256
OWNERS = ("self", "opponent", "neutral")


def integer(value: int, minimum: int, maximum: int, name: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}: expected integer {minimum}..{maximum}")


def position(value: tuple[int, int, int]) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Position must be an immutable (x, y, z) tuple")
    for coordinate, extent in zip(value, BOARD_SIZE):
        integer(coordinate, 0, extent - 1, "position")


@dataclass(frozen=True)
class Tile:
    position: tuple[int, int, int]
    terrain: int
    blocked: bool
    visitable: bool

    def __post_init__(self):
        position(self.position)
        integer(self.terrain, 0, TERRAIN_TYPES - 1, "terrain")
        if type(self.blocked) is not bool or type(self.visitable) is not bool:
            raise ValueError("Tile flags must be boolean")


@dataclass(frozen=True)
class ObjectView:
    # Bookkeeping only. The encoder must be invariant to handle renaming.
    handle: str
    position: tuple[int, int, int]
    kind: int
    subtype: int
    owner: Literal["self", "opponent", "neutral"]

    def __post_init__(self):
        if type(self.handle) is not str or not self.handle:
            raise ValueError("An observed object needs an opaque local handle")
        position(self.position)
        integer(self.kind, 0, OBJECT_TYPES - 1, "object kind")
        integer(self.subtype, 0, OBJECT_SUBTYPES - 1, "object subtype")
        if self.owner not in OWNERS:
            raise ValueError("Object owner must be relative to the observing player")


@dataclass(frozen=True)
class Stack:
    slot: int
    creature: int
    count: int

    def __post_init__(self):
        integer(self.slot, 0, 6, "army slot")
        integer(self.creature, 0, CREATURE_TYPES - 1, "creature")
        integer(self.count, 1, 2**31 - 1, "stack count")


@dataclass(frozen=True)
class OwnState:
    hero: tuple[int, int, int] | None
    movement: int
    movement_max: int
    resources: tuple[int, ...]
    army: tuple[Stack, ...]
    towns: tuple[tuple[int, int, int], ...]
    mines: int

    def __post_init__(self):
        if self.hero is not None:
            position(self.hero)
        for name in ("movement", "movement_max", "mines"):
            integer(getattr(self, name), 0, 2**31 - 1, name)
        if not isinstance(self.resources, tuple) or len(self.resources) != 7:
            raise ValueError("Own resources must contain the seven VCMI resource amounts")
        for amount in self.resources:
            integer(amount, 0, 2**31 - 1, "resource amount")
        if not isinstance(self.army, tuple) or any(type(s) is not Stack for s in self.army):
            raise ValueError("Own army must contain Stack records")
        if len({s.slot for s in self.army}) != len(self.army):
            raise ValueError("Duplicate army slot")
        if self.hero is None and (self.army or self.movement or self.movement_max):
            raise ValueError("A missing hero cannot carry army or movement")
        if not isinstance(self.towns, tuple):
            raise ValueError("Own towns must be an immutable tuple")
        for town in self.towns:
            position(town)
        if len(set(self.towns)) != len(self.towns):
            raise ValueError("Duplicate own town position")


@dataclass(frozen=True)
class View:
    tiles: tuple[Tile, ...]
    objects: tuple[ObjectView, ...]
    own: OwnState
    size: tuple[int, int, int] = BOARD_SIZE

    def __post_init__(self):
        if self.size != BOARD_SIZE or type(self.size) is not tuple:
            raise ValueError("Prototype supports only the authored 20x20x1 scenario")
        if type(self.own) is not OwnState:
            raise ValueError("View requires explicit own state")
        if not isinstance(self.tiles, tuple) or any(type(t) is not Tile for t in self.tiles):
            raise ValueError("View requires immutable visible Tile records")
        locations = {t.position for t in self.tiles}
        if len(locations) != len(self.tiles):
            raise ValueError("Duplicate visible tile")
        if not isinstance(self.objects, tuple) or any(type(o) is not ObjectView for o in self.objects):
            raise ValueError("View requires immutable current ObjectView records")
        if len({o.handle for o in self.objects}) != len(self.objects):
            raise ValueError("Duplicate current object handle")
        if any(o.position not in locations for o in self.objects):
            raise ValueError("Current object position must be visible in this view")


@dataclass(frozen=True)
class Frame:
    episode: str
    player: int
    seq: int
    day: int
    kind: Literal["view", "decision", "action", "result"]
    view: View | None = None
    action: int | None = None
    accepted: bool | None = None
    mask: tuple[bool, ...] | None = None
    decision_seq: int | None = None
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError("Unsupported learner history schema; raw bridge v2 is not supported")
        if type(self.episode) is not str or not self.episode:
            raise ValueError("Missing episode identity")
        integer(self.player, 0, 1, "player")
        integer(self.seq, 0, 2**63 - 1, "player history sequence")
        integer(self.day, 1, 2**31 - 1, "public game day")
        if self.kind not in ("view", "decision", "action", "result"):
            raise ValueError("Unsupported history frame kind")
        if self.kind in ("view", "decision"):
            if (type(self.view) is not View or self.action is not None
                    or self.accepted is not None or self.decision_seq is not None):
                raise ValueError("Observation frame requires only a current player view")
        elif self.view is not None or self.mask is not None:
            raise ValueError("Own action/result frames must not carry a view or mask")
        else:
            integer(self.action, 0, 8, "own action")
            integer(self.decision_seq, 0, self.seq - 1, "referenced decision sequence")
            if self.kind == "result" and type(self.accepted) is not bool:
                raise ValueError("Result requires an explicit acceptance flag")
            if self.kind == "action" and self.accepted is not None:
                raise ValueError("Own action cannot anticipate its result")
        if self.kind == "decision":
            if (not isinstance(self.mask, tuple) or len(self.mask) != 9
                    or any(type(v) is not bool for v in self.mask) or not self.mask[8]):
                raise ValueError("Decision requires nine boolean attempts with end-turn enabled")
            if self.view.own.hero is None and any(self.mask[:8]):
                raise ValueError("Town-only state permits end-turn only")
        elif self.mask is not None:
            raise ValueError("Only a decision carries an attempt mask")


class HistoryCursor:
    """One player's stream; validates chronology without interpreting hidden state."""

    def __init__(self, episode: str, player: int):
        if type(episode) is not str or not episode:
            raise ValueError("Missing episode identity")
        integer(player, 0, 1, "player")
        self.episode, self.player = episode, player
        self.next_seq, self.day = 0, 1
        self.decision: Frame | None = None
        self.pending_action: tuple[int, int] | None = None

    def consume(self, frame: Frame) -> None:
        if type(frame) is not Frame:
            raise ValueError("Expected typed learner Frame, never a raw bridge message")
        if (frame.episode, frame.player) != (self.episode, self.player):
            raise ValueError("Cross-player or cross-episode history")
        if frame.seq != self.next_seq or frame.day < self.day:
            raise ValueError("History is missing, duplicated, reordered or has a decreasing day")
        if frame.kind == "decision":
            if self.decision is not None or self.pending_action is not None:
                raise ValueError("New decision before prior action/result completed")
            self.decision = frame
        elif frame.kind == "action":
            if (self.decision is None or self.decision.seq != frame.decision_seq
                    or not self.decision.mask[frame.action]):
                raise ValueError("Own action must follow its decision and permitted attempt mask")
            self.decision = None
            self.pending_action = (frame.decision_seq, frame.action)
        elif frame.kind == "result":
            if self.pending_action != (frame.decision_seq, frame.action):
                raise ValueError("Result must match the pending own action")
            self.pending_action = None
        elif self.decision is not None:
            raise ValueError("View changed after decision; obtain a new atomic decision instead")
        self.next_seq += 1
        self.day = frame.day


def _fields(value: dict, required: set[str], optional: set[str] = frozenset()) -> dict:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("Missing or unrecognized fields in player-history record")
    return dict(value)


def _array(value, name: str) -> tuple:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def decode_frame(value: dict) -> Frame:
    """Decode an allowlisted wire record; reject raw v2 and arbitrary engine fields."""
    data = _fields(value, {"schema", "episode", "player", "seq", "day", "kind"},
                   {"view", "action", "accepted", "mask", "decision_seq"})
    if data.get("mask") is not None:
        data["mask"] = _array(data["mask"], "mask")
    if data.get("view") is not None:
        view = _fields(data["view"], {"size", "tiles", "objects", "own"})
        view["size"] = _array(view["size"], "map size")
        tiles = []
        for raw in _array(view["tiles"], "tiles"):
            tile = _fields(raw, {"position", "terrain", "blocked", "visitable"})
            tile["position"] = _array(tile["position"], "tile position")
            tiles.append(Tile(**tile))
        objects = []
        for raw in _array(view["objects"], "objects"):
            obj = _fields(raw, {"handle", "position", "kind", "subtype", "owner"})
            obj["position"] = _array(obj["position"], "object position")
            objects.append(ObjectView(**obj))
        own = _fields(view["own"], {"hero", "movement", "movement_max", "resources", "army", "towns", "mines"})
        if own["hero"] is not None:
            own["hero"] = _array(own["hero"], "own hero position")
        own["resources"] = _array(own["resources"], "resources")
        own["army"] = tuple(Stack(**_fields(s, {"slot", "creature", "count"}))
                            for s in _array(own["army"], "army"))
        own["towns"] = tuple(_array(p, "town position") for p in _array(own["towns"], "towns"))
        data["view"] = View(tuple(tiles), tuple(objects), OwnState(**own), view["size"])
    return Frame(**data)
