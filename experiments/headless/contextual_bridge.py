"""One player's adventure-windows-v1 bridge stream (protocol version 4).

A ``ContextualPlayer`` validates every native message of one player in arrival
order and turns it into that player's heroes-player-history-v2 frames, using
the strict contract in ``experiments/learner/contextual.py``. It folds the
ordered context updates into the full player-visible observation each view and
decision carries, replaces engine IDs with player-local handles assigned in
first-seen order, and makes every owner relative. Engine IDs stay in the raw
bridge evidence and in the native responses built here; they never reach a
frame. Any violation is a technical failure of the episode.

After a provisional cutoff the learner prefix is frozen: later messages are
still validated and folded, so continuation answers and late failures remain
checked, but they are raw evidence only and never become frames.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.learner import contextual as contract  # noqa: E402

from bridge_protocol import ProtocolError  # noqa: E402

PROTOCOL_VERSION = 4
READY = {"interface": contract.ENVIRONMENT, "action_schema": contract.ACTION_SCHEMA,
         "observation_events": "context_update-v1", "routing": "player-knowledge-current-turn-v1",
         "notices": "audited-notice-v1"}
ENVELOPE = {"type", "protocol_version", "player", "episode"}
RESOURCES = 7
NEUTRAL = 255
PLAYER_COLORS = 8
UNSUPPORTED_PREFIX = "Unsupported scenario interaction"


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _integer(value, low=None):
    return type(value) is int and (low is None or value >= low)


def _fields(message, required, optional=(), what="message"):
    _require(isinstance(message, dict), f"Expected a {what} object")
    keys = set(message)
    _require(set(required) <= keys and keys <= set(required) | set(optional), f"Unexpected {what} fields: {sorted(keys)}")


def _position(value):
    _require(isinstance(value, list) and len(value) == 3 and all(_integer(x, 0) for x in value), "Malformed position")
    return tuple(value)


class ContextualPlayer:
    """Validator, observation fold and history-v2 producer of one player."""

    def __init__(self, player: int, color: str, episode: str):
        self.player, self.color, self.episode = player, color, episode
        self.history = contract.PlayerHistory(episode, player)
        self.encoded = []  # JSON form of each frame, made once
        self.texts = []  # the same frames as canonical JSON lines
        self.ready = False
        self.size = None
        self.snapshot_every = None
        self.terminal = None
        self.frozen = False  # the learner prefix ended at a provisional cutoff
        self.next_sequence = 0
        self.next_decision_id = 0
        self.revision = 0  # the last native revision of a decision or context change
        self.handles = {}  # engine ID -> handle; bookkeeping only
        self.engine_ids = {}
        self.tiles = {}  # position -> (terrain, road, blocked, visitable)
        self.objects = {}  # engine ID -> native visible object
        self.own = None  # native own state
        self.window = None  # the open contract.Window
        self.native_window = None  # the open window's native fields, until its context event
        self.decision = None  # the held native decision
        self.awaiting = False  # a response was delivered for the held decision
        self.day = 1
        self._observation_cache = None

    # -- native stream ---------------------------------------------------------
    def check(self, message) -> str:
        """Validate and fold one native message; return its type."""
        _require(isinstance(message, dict), "Bridge message must be an object")
        kind = message.get("type")
        _require(message.get("protocol_version") == PROTOCOL_VERSION, "Unsupported bridge protocol version")
        _require(message.get("player") == self.player and message.get("episode") == self.episode,
                 "Message names another player or episode")
        if kind == "error":
            _fields(message, ENVELOPE | {"message"})
            _require(type(message["message"]) is str, "Engine error requires a message")
            return kind
        _require(self.terminal is None, "Message after this player's terminal")
        if kind == "ready":
            self._ready(message)
            return kind
        _require(self.ready, "Message before ready")
        if kind == "decision":
            self._decision(message)
        elif kind == "result":
            self._result(message)
        elif kind == "terminal":
            self._terminal(message)
        else:
            raise ProtocolError(f"Unsupported contextual message type {kind!r}")
        return kind

    def _ready(self, message):
        _fields(message, ENVELOPE | set(READY) | {"battle_ai", "profiling", "snapshot_every", "map_size"})
        _require(not self.ready, "Duplicate ready")
        for key, value in READY.items():
            _require(message[key] == value, f"Unsupported contextual {key}")
        _require(type(message["profiling"]) is bool and type(message["battle_ai"]) is str, "Malformed ready")
        _require(_integer(message["snapshot_every"], 0), "Malformed snapshot interval")
        size = _position(message["map_size"])
        _require(min(size) > 0 and size[2] == 1, "The contextual interface supports one map level")
        self.size, self.snapshot_every, self.ready = size, message["snapshot_every"], True

    def _decision(self, message):
        _fields(message, ENVELOPE | {"decision_id", "revision", "context", "day", "events"}, ("state", "perf"))
        _require(self.decision is None, "Decision while another is outstanding")
        _require(message["decision_id"] == self.next_decision_id, "Decision identifiers must be consecutive")
        _require(_integer(message["revision"]) and message["revision"] > self.revision, "Decision revision must advance")
        _require(_integer(message["day"], 1), "Malformed decision day")
        self._events(message["events"], acting=False)
        _require(self._native_context(message["context"]) == self._context(), "Decision context disagrees with the history")
        if "state" in message:
            self._verify_state(message["state"])
        _require(self.own is not None, "Decision before the first context capture")
        observation = self._observation()
        self.revision, self.next_decision_id = message["revision"], self.next_decision_id + 1
        offers = contract.Offers.from_observation(observation)
        frame = self._emit("decision", day=message["day"], observation=observation, offers=offers,
                           revision=message["revision"])
        self.decision = {"id": message["decision_id"], "revision": message["revision"], "offers": offers,
                         "seq": frame.seq if frame is not None else None, "day": message["day"]}
        self.awaiting = False

    def _result(self, message):
        _fields(message, ENVELOPE | {"decision_id", "events", "result"}, ("window", "perf"))
        _require(self.decision is not None and self.awaiting and message["decision_id"] == self.decision["id"],
                 "Result without its delivered decision")
        self.native_window = message.get("window")
        self._events(message["events"], acting=True)
        _require(self.native_window is None, "Window fields without an opened window context")
        result = self._native_result(message["result"])
        self._emit("result", result=result, decision_seq=self.decision["seq"])
        self.decision, self.awaiting = None, False

    def _terminal(self, message):
        _fields(message, ENVELOPE | {"outcome", "events"}, ("result",))
        _require(message["outcome"] in ("win", "loss"), "Unsupported terminal outcome")
        if self.decision is not None and not self.awaiting:
            # Never answered or never delivered: withdrawn before anything visible follows.
            self._emit("superseded", decision_seq=self.decision["seq"], reason="terminal")
            self.decision = None
        acting = self.decision is not None
        self._events(message["events"], acting=acting)
        if "result" in message:
            _require(acting, "Terminal result without an action in flight")
            self._emit("result", result=self._native_result(message["result"]), decision_seq=self.decision["seq"])
        elif acting:
            # Consumed or not, its outcome is unknown: the terminal supersedes it.
            self._emit("superseded", decision_seq=self.decision["seq"], reason="terminal")
        self.decision, self.awaiting = None, False
        self._emit("terminal", outcome=message["outcome"])
        if self.window is not None:
            self._emit("context_invalidated", context=contract.Context(), revision=self.history.revision + 1,
                       reason="terminal")
            self.window = None
        self.terminal = message["outcome"]

    def _events(self, events, acting):
        _require(isinstance(events, list), "Events must be a list")
        for event in events:
            _require(isinstance(event, dict), "Malformed event")
            kind = event.get("kind")
            if kind == "context_update":
                self._fold(event)
                self._emit("view", day=event["day"], observation=self._observation())
            elif kind == "visit":
                _fields(event, {"kind", "object_id", "object_type", "object_subtype", "owner"}, what="visit")
                _require(all(_integer(event[key]) for key in ("object_id", "object_type", "object_subtype", "owner")),
                         "Malformed visit")
                self._emit_event(contract.VisibleEvent("visit", subject=self._handle(event["object_id"]),
                                                       object_kind=event["object_type"], object_subtype=event["object_subtype"],
                                                       owner=self._owner(event["owner"])))
            elif kind == "battle_start":
                _fields(event, {"kind", "side"}, what="battle start")
                self._emit_event(contract.VisibleEvent("battle_start", side=event["side"]))
            elif kind == "battle_complete":
                _fields(event, {"kind", "won"}, what="battle completion")
                self._emit_event(contract.VisibleEvent("battle_complete", won=event["won"]))
            elif kind == "notice":
                _fields(event, {"kind", "notice"}, what="notice")
                self._emit_event(contract.VisibleEvent("notice", notice=contract.decode_notice(event["notice"])))
            elif kind == "hero_lost":
                _fields(event, {"kind", "hero_id"}, what="hero loss")
                _require(event["hero_id"] in self.handles, "Lost hero was never observed")
                self._emit_event(contract.VisibleEvent("hero_lost", subject=self.handles[event["hero_id"]], owner="self"))
            elif kind == "turn_end":
                _fields(event, {"kind"}, what="turn end")
                self._emit_event(contract.VisibleEvent("turn_end"))
            elif kind in ("consumed", "issued", "acknowledged", "context"):
                _require(acting, f"{kind} event without an action in flight")
                self._action_event(event)
            else:
                raise ProtocolError(f"Unsupported contextual event {kind!r}")

    def _action_event(self, event):
        seq = self.decision["seq"]
        kind = event["kind"]
        if kind == "consumed":
            _fields(event, {"kind", "consumer"}, what="consumption")
            self._emit("consumed", consumer=event["consumer"], decision_seq=seq)
        elif kind == "issued":
            _fields(event, {"kind", "command_index"}, what="issue")
            self._emit("issued", command_index=event["command_index"], decision_seq=seq)
        elif kind == "acknowledged":
            _fields(event, {"kind", "command_index", "accepted"}, what="acknowledgement")
            self._emit("acknowledged", command_index=event["command_index"], accepted=event["accepted"], decision_seq=seq)
        else:
            _fields(event, {"kind", "context", "revision"}, what="context change")
            _require(_integer(event["revision"]) and event["revision"] > self.revision, "Context revision must advance")
            context = self._native_context(event["context"])
            if context.kind == "map":
                self.window = None
            else:
                self.window = self._native_window_fields(context)
            self.revision = event["revision"]
            self._observation_cache = None
            self._emit("context", context=context, revision=event["revision"], decision_seq=seq)

    # -- fold ------------------------------------------------------------------
    def _fold(self, event):
        _fields(event, {"kind", "sequence", "day", "appeared", "changed", "disappeared", "revealed", "tiles_changed",
                        "hidden", "own"}, what="context update")
        _require(event["sequence"] == self.next_sequence, "Context updates must be consecutive")
        _require(_integer(event["day"], 1), "Malformed context update day")
        self.next_sequence += 1
        for name in ("appeared", "changed", "disappeared", "revealed", "tiles_changed", "hidden"):
            _require(isinstance(event[name], list), f"Malformed {name}")
        for position in event["hidden"]:
            position = _position(position)
            _require(position in self.tiles, "Hidden tile was not visible")
            del self.tiles[position]
        for name, known in (("revealed", False), ("tiles_changed", True)):
            for tile in event[name]:
                _fields(tile, {"x", "y", "z", "terrain", "road", "blocked", "visitable"}, what="tile")
                position = self._in_bounds((tile["x"], tile["y"], tile["z"]))
                _require((position in self.tiles) == known, f"Inconsistent {name} tile")
                _require(_integer(tile["terrain"], 0) and _integer(tile["road"], 0)
                         and type(tile["blocked"]) is bool and type(tile["visitable"]) is bool, "Malformed tile facts")
                self.tiles[position] = (tile["terrain"], tile["road"], tile["blocked"], tile["visitable"])
        for identifier in event["disappeared"]:
            _require(identifier in self.objects, "Unknown object disappeared")
            del self.objects[identifier]
        for name, known in (("appeared", False), ("changed", True)):
            for entry in event[name]:
                _fields(entry, {"id", "x", "y", "z", "type", "subtype", "owner"}, what="object")
                _require(all(_integer(entry[key]) for key in entry), "Malformed object")
                _require((entry["id"] in self.objects) == known, f"Inconsistent {name} object")
                self._in_bounds((entry["x"], entry["y"], entry["z"]))
                self.objects[entry["id"]] = dict(entry)
                self._handle(entry["id"])
        if event["own"] is not None:
            self.own = self._native_own(event["own"])
        self.day = max(self.day, event["day"])
        self._observation_cache = None

    def _native_own(self, own):
        _fields(own, {"resources", "heroes", "towns"}, what="own state")
        _require(isinstance(own["resources"], list) and len(own["resources"]) == RESOURCES
                 and all(_integer(x, 0) for x in own["resources"]), "Malformed resources")
        for name, keys in (("heroes", {"id", "position", "movement", "movement_max"}), ("towns", {"id", "position"})):
            _require(isinstance(own[name], list), f"Malformed own {name}")
            for entry in own[name]:
                _fields(entry, keys, what=f"own {name}")
                _require(_integer(entry["id"], 0), "Malformed own identifier")
                self._in_bounds(_position(entry["position"]))
                if name == "heroes":
                    _require(_integer(entry["movement"], 0) and _integer(entry["movement_max"], 0), "Malformed movement")
                self._handle(entry["id"])
        return own

    def _verify_state(self, state):
        # The periodic capture repeats what the events established: the fold must match it exactly.
        _fields(state, {"tiles", "objects", "own"}, what="state")
        tiles = {}
        for tile in state["tiles"]:
            tiles[(tile["x"], tile["y"], tile["z"])] = (tile["terrain"], tile["road"], tile["blocked"], tile["visitable"])
        objects = {entry["id"]: entry for entry in state["objects"]}
        _require(tiles == self.tiles and objects == self.objects and state["own"] == self.own,
                 "Context update fold disagrees with the decision state")

    # -- player-visible forms --------------------------------------------------
    def _handle(self, identifier):
        _require(_integer(identifier, 0), "Malformed engine identifier")
        if identifier not in self.handles:
            handle = f"object-{len(self.handles) + 1}"
            self.handles[identifier], self.engine_ids[handle] = handle, identifier
        return self.handles[identifier]

    def _owner(self, owner):
        _require(_integer(owner), "Malformed owner")
        if owner == NEUTRAL:
            return "neutral"
        _require(0 <= owner < PLAYER_COLORS, "Unsupported owner")
        return "self" if owner == self.player else "opponent"

    def _in_bounds(self, position):
        _require(all(_integer(x, 0) for x in position) and all(p < n for p, n in zip(position, self.size)),
                 "Position outside the public map bounds")
        return tuple(position)

    def _context(self):
        return self.window.context if self.window is not None else contract.Context()

    def _native_context(self, context):
        _fields(context, {"kind", "subject"}, what="context")
        if context["kind"] == "map":
            _require(context["subject"] is None, "Map context has no subject")
            return contract.Context()
        _require(context["kind"] in ("hero", "town") and context["subject"] in self.handles, "Unknown context subject")
        return contract.Context(context["kind"], self.handles[context["subject"]])

    def _native_window_fields(self, context):
        window, self.native_window = self.native_window, None
        _require(window is not None, "Opened context without its window fields")
        if context.kind == "hero":
            _fields(window, {"kind", "subject", "army", "primary_skills", "mana", "level"}, what="hero window")
            _require(isinstance(window["primary_skills"], list), "Malformed primary skills")
            fields = {"primary_skills": tuple(window["primary_skills"]), "mana": window["mana"], "level": window["level"]}
        else:
            _fields(window, {"kind", "subject", "army", "faction"}, what="town window")
            fields = {"faction": window["faction"]}
        _require(isinstance(window["army"], list), "Malformed window army")
        for stack in window["army"]:
            _fields(stack, {"slot", "creature", "count"}, what="army stack")
        army = tuple(contract.Stack(stack["slot"], stack["creature"], stack["count"]) for stack in window["army"])
        _require(window["kind"] == context.kind and self.handles.get(window["subject"]) == context.subject,
                 "Window fields describe another subject")
        return contract.Window(context, army, **fields)

    def _native_result(self, result):
        _fields(result, {"status", "accepted"}, ("position", "movement", "destination_reached"), what="result")
        values = dict(result)
        if "position" in values:
            values["position"] = _position(values["position"])
        return contract.Result(**values)

    def _observation(self):
        if self._observation_cache is None:
            order = lambda position: (position[2], position[1], position[0])  # noqa: E731
            number = lambda handle: int(handle.split("-")[1])  # noqa: E731
            tiles = tuple(contract.Tile(position, *facts) for position, facts in sorted(self.tiles.items(), key=lambda t: order(t[0])))
            objects = tuple(sorted((contract.ObjectView(self.handles[i], (o["x"], o["y"], o["z"]), o["type"], o["subtype"],
                                                        self._owner(o["owner"])) for i, o in self.objects.items()),
                                   key=lambda view: number(view.handle)))
            own = self.own
            heroes = tuple(sorted((contract.Hero(self.handles[h["id"]], tuple(h["position"]), h["movement"], h["movement_max"])
                                   for h in own["heroes"]), key=lambda hero: number(hero.handle)))
            towns = tuple(sorted((contract.Town(self.handles[t["id"]], tuple(t["position"])) for t in own["towns"]),
                                 key=lambda town: number(town.handle)))
            self._observation_cache = contract.Observation(self.size, tiles, objects, tuple(own["resources"]), heroes, towns,
                                                           self.window)
        return self._observation_cache

    # -- history ---------------------------------------------------------------
    def _emit(self, kind, day=None, **payload):
        if self.frozen:
            return None
        return self.history.emit(kind, day=day, **payload)

    def _emit_event(self, event):
        self._emit("event", event=event)

    def frames_json(self, start: int = 0) -> list[dict]:
        """The JSON form of frames from ``start``; encodings are made once and shared, so never mutate them."""
        frames = self.history.frames
        for frame in frames[len(self.encoded):]:
            encoded = contract.encode_frame(frame)
            self.encoded.append(encoded)
            self.texts.append(json.dumps(encoded, sort_keys=True, separators=(",", ":")))
        return self.encoded[start:]

    def frames_text(self) -> list[str]:
        """Each frame as one canonical JSON line (sorted keys, compact), as the history file stores it."""
        self.frames_json(len(self.encoded))
        return self.texts

    # -- responses -------------------------------------------------------------
    def open_decision(self) -> bool:
        return self.decision is not None and not self.awaiting

    def respond(self, action) -> dict:
        """The native response to the held decision; raises ValueError and changes nothing when not offered."""
        if not self.open_decision() or self.frozen:
            raise ValueError("No open contextual decision for this player")
        if type(action) is not contract.Action or not self.decision["offers"].permits(action):
            raise ValueError("Action is not offered at this decision")
        native = {"kind": action.kind}
        if action.subject is not None:
            native["subject"] = self.engine_ids[action.subject]
        if action.kind == "move_to":
            native["x"], native["y"] = action.x, action.y
        return {"decision_id": self.decision["id"], "revision": self.decision["revision"], "action": native}

    def submitted(self, action) -> None:
        """Records a delivered response."""
        self._emit("submitted", action=action, decision_seq=self.decision["seq"])
        self.awaiting = True

    def continuation(self) -> dict:
        """Raw-evidence answer after the cutoff: return to the map, then end the turn."""
        _require(self.frozen and self.open_decision(), "Continuation needs a held decision after the cutoff")
        kind = "close" if self.window is not None else "end_turn"
        self.awaiting = True
        return {"decision_id": self.decision["id"], "revision": self.decision["revision"], "action": {"kind": kind}}

    def cut(self, reason: str) -> int:
        """Marks the held, unanswered decision as this player's provisional cutoff and freezes the prefix."""
        _require(self.open_decision() and not self.frozen, "A cutoff needs this player's real unacted decision")
        self._emit("cutoff", decision_seq=self.decision["seq"], reason=reason)
        self.frozen = True
        return self.decision["seq"]

    def fail(self, reason: str) -> None:
        """Ends the history as a failure, whatever its state; a finalized history stays as it is."""
        if self.history.boundary in (None, "cutoff_pending"):
            self.history.emit("failure", reason=reason)
