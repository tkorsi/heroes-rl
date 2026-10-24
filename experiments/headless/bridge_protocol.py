"""Validation of one player's ExplorerRL bridge stream (protocol version 3).

Every engine message for one player passes through a ``BridgeValidator`` in
arrival order. It enforces identity, ordering and the shape of the
player-visible event stream; it never interprets hidden state, because the
stream contains none. Any violation is a technical failure of the episode.
"""
from __future__ import annotations

PROTOCOL_VERSION = 3
OBSERVATION_EVENTS = "visible_update-v2"
ACTIONS = 9
END_TURN = 8
EVENT_KINDS = {"visible_update", "battle_start", "battle_complete", "visit", "decision"}
OBJECT_FIELDS = ("id", "x", "y", "z", "type", "subtype", "owner")
TILE_FIELDS = ("x", "y", "z", "terrain", "blocked", "visitable")
OWN_FIELDS = ("hero", "movement", "movement_max", "resources", "army", "towns", "mines")
RESOURCES = 7


class ProtocolError(ValueError):
    """The engine stream violated the bridge contract."""


def _integer(value) -> bool:
    return type(value) is int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_mask(mask) -> list[bool]:
    _require(isinstance(mask, list) and len(mask) == ACTIONS and all(type(x) is bool for x in mask) and mask[END_TURN],
             "Expected nine boolean attempt-mask entries with end-turn enabled")
    return mask


def validate_choices(choices) -> list[int]:
    _require(isinstance(choices, list) and bool(choices)
             and all(isinstance(x, dict) and _integer(x.get("choice")) for x in choices),
             "Decision must provide explicit integer choices")
    values = [x["choice"] for x in choices]
    _require(len(set(values)) == len(values), "Duplicate decision choices")
    return values


def _validate_object(entry) -> None:
    _require(isinstance(entry, dict) and set(OBJECT_FIELDS) <= set(entry)
             and all(_integer(entry[key]) for key in OBJECT_FIELDS), "Malformed visible object")


def _validate_position(tile) -> None:
    _require(isinstance(tile, list) and len(tile) == 3 and all(_integer(x) for x in tile), "Malformed tile coordinate")


def _validate_tile_fact(tile) -> None:
    _require(isinstance(tile, dict) and set(tile) == set(TILE_FIELDS)
             and all(_integer(tile[key]) for key in ("x", "y", "z", "terrain"))
             and type(tile["blocked"]) is bool and type(tile["visitable"]) is bool, "Malformed visible tile facts")


def validate_own(own) -> None:
    """The complete own state carried by a visible update when it changed."""
    _require(isinstance(own, dict) and set(own) == set(OWN_FIELDS), "Malformed own state")
    if own["hero"] is not None:
        _validate_position(own["hero"])
    _require(all(_integer(own[key]) and own[key] >= 0 for key in ("movement", "movement_max", "mines")),
             "Own movement and mine count must be non-negative integers")
    _require(isinstance(own["resources"], list) and len(own["resources"]) == RESOURCES
             and all(_integer(x) for x in own["resources"]), "Own state must list the seven resource amounts")
    _require(isinstance(own["army"], list) and all(
        isinstance(stack, dict) and set(stack) == {"slot", "creature", "count"} and all(_integer(v) for v in stack.values())
        for stack in own["army"]), "Malformed own army")
    _require(isinstance(own["towns"], list), "Own towns must be a list")
    for town in own["towns"]:
        _validate_position(town)


class BridgeValidator:
    """Ordering and shape checks for one player's engine messages."""

    def __init__(self, player: int, episode: str):
        self.player, self.episode = player, episode
        self.ready = False
        self.terminal = None
        self.last_action_id = -1
        self.last_decision_id = -1
        self.pending = None  # ("action", action_id, mask) or ("decision", decision_id, choices)
        self.submitted = None  # (action_id, action) awaiting its result
        self.awaiting_turn_end = False
        self.last_day = 0
        self.next_visible_sequence = 0

    def check(self, message) -> str:
        """Validate the next engine message and return its type."""
        _require(isinstance(message, dict), "Bridge message must be an object")
        _require(_integer(message.get("player")) and message["player"] == self.player
                 and message.get("episode") == self.episode, "Bridge message has wrong episode/player identity")
        _require(message.get("protocol_version") == PROTOCOL_VERSION, "Unsupported bridge protocol version")
        kind = message.get("type")
        _require(self.terminal is None, "Message received after terminal")
        if kind == "ready":
            _require(not self.ready, "Duplicate bridge handshake")
            _require(message.get("adventure_actions") == ACTIONS and message.get("observation_events") == OBSERVATION_EVENTS,
                     "Bridge does not provide the required action interface and visible event stream")
            size = message.get("map_size")
            _require(isinstance(size, list) and len(size) == 3 and all(_integer(x) and x > 0 for x in size),
                     "Bridge handshake must state the public map size")
            self.ready = True
            return kind
        _require(self.ready, "Bridge must send its handshake before any game message")
        if kind == "error":
            return kind  # reported by the engine; the caller fails the episode
        if kind in {"observation", "action_result", "decision", "terminal"}:
            self._check_events(message.get("events"))
        if kind == "observation":
            action_id, day = message.get("action_id"), message.get("day")
            _require(_integer(action_id) and action_id == self.last_action_id + 1 and self.pending is None
                     and self.submitted is None and not self.awaiting_turn_end,
                     "Observation action sequence is incomplete or out of order")
            _require(_integer(day) and day >= max(self.last_day, 1), "Observation game day is invalid")
            _require(isinstance(message.get("tiles"), list), "Observation must contain player-visible tiles")
            _require(isinstance(message.get("state"), dict), "Observation must contain the player's own state")
            self.last_day = day
            self.last_action_id = action_id
            self.pending = ("action", action_id, validate_mask(message.get("action_mask")))
        elif kind == "action_result":
            _require(self.submitted is not None and (message.get("action_id"), message.get("action")) == self.submitted,
                     "Action result does not match the submitted action")
            _require(type(message.get("accepted")) is bool, "Action result is missing an acceptance flag")
            _require(isinstance(message.get("after"), dict) and isinstance(message.get("before"), dict),
                     "Action result must contain before/after player state")
            self.awaiting_turn_end = self.submitted[1] == END_TURN and message["accepted"]
            self.submitted = None
        elif kind == "decision":
            decision_id = message.get("decision_id")
            _require(_integer(decision_id) and decision_id == self.last_decision_id + 1, "Decision ID must increase by one")
            _require(self.pending is None, "Decision arrived while another response is outstanding")
            self.last_decision_id = decision_id
            self.pending = ("decision", decision_id, validate_choices(message.get("choices")))
        elif kind == "turn_end":
            _require(self.awaiting_turn_end and self.submitted is None, "Turn end must follow an accepted end-turn action")
            self.awaiting_turn_end = False
        elif kind == "terminal":
            _require(message.get("outcome") in {"win", "loss"}, "Terminal must contain a natural win or loss")
            _require(isinstance(message.get("state"), dict), "Terminal must contain the player's final state")
            self.terminal = message["outcome"]
            self.pending = None  # an unanswered request is void after the game ended
        else:
            raise ProtocolError(f"Unsupported bridge message: {kind}")
        return kind

    def respond(self, value: int) -> dict:
        """Check a response to the outstanding request and return its wire form."""
        if self.pending is None:
            raise ValueError("No outstanding request for this player")
        kind, identifier, allowed = self.pending
        if kind == "action":
            if type(value) is not int or not 0 <= value < ACTIONS or not allowed[value]:
                raise ValueError("Action is outside the attempt mask")
            self.pending, self.submitted = None, (identifier, value)
            return {"action_id": identifier, "action": value}
        if type(value) is not int or value not in allowed:
            raise ValueError("Choice is not offered by the decision")
        self.pending = None
        return {"decision_id": identifier, "choice": value}

    def _check_events(self, events) -> None:
        _require(isinstance(events, list), "Message must carry its ordered event list")
        for event in events:
            _require(isinstance(event, dict) and event.get("kind") in EVENT_KINDS, "Unknown or malformed event")
            if event["kind"] != "visible_update":
                continue
            _require(event.get("sequence") == self.next_visible_sequence, "Visible update sequence has a gap")
            self.next_visible_sequence += 1
            for key in ("appeared", "changed"):
                _require(isinstance(event.get(key), list), "Visible update lists are required")
                for entry in event[key]:
                    _validate_object(entry)
            _require(isinstance(event.get("disappeared"), list) and all(_integer(x) for x in event["disappeared"]),
                     "Visible update disappearance list is malformed")
            for key in ("revealed", "tiles_changed", "hidden"):
                _require(isinstance(event.get(key), list), "Visible update tile lists are required")
            for tile in event["revealed"] + event["tiles_changed"]:
                _validate_tile_fact(tile)
            for tile in event["hidden"]:
                _validate_position(tile)
            _require("own" in event, "Visible update must state whether the own state changed")
            if event["own"] is not None:
                validate_own(event["own"])
