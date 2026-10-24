"""One player's sanitized, ordered learner history.

The engine stream carries engine object IDs only so that this adapter can route
facts. They never leave it: each player gets opaque, episode-local handles,
assigned in order of that player's own first legitimate observation. Handles
are never created for an object that this player has not seen, so a handle
cannot leak the existence of a hidden object. Two players' histories share no
state; the same engine object normally has different handles for each.

Visible updates are folded into the player's current view (tiles with their
terrain and flags, objects, and the player's own state) and a last-seen
record. A visible object is seen again at every capture and decision, so its
last-seen time follows them. A disappearance only marks a remembered fact as no
longer visible: its position, attributes and time stay those of the last
observation and never refresh from hidden state. At every decision snapshot
the folded view must equal the snapshot's visible tiles, objects and own state,
which proves that the event stream alone reconstructs what this player saw
between decisions.

The folded view is also projected onto the learner's player-history v1 frames
(see ``history_v1.py``). Frame substructures are shared between frames and are
replaced, never modified, when a fact changes; consumers must not mutate them.
"""
from __future__ import annotations

import copy

from bridge_protocol import ProtocolError
from history_v1 import HistoryV1, relative_owner

FACT_FIELDS = ("x", "y", "z", "type", "subtype", "owner")
TILE_FACTS = ("terrain", "blocked", "visitable")
DECISION_FIELDS = {"kind", "decision_id", "primary_skill", "text", "safe_to_autoaccept", "choices"}
CHOICE_FIELDS = {"choice", "secondary_skill", "component_type", "component_subtype", "value"}


def _position_key(position: tuple[int, int, int]) -> tuple[int, int, int]:
    return position[2], position[1], position[0]  # the engine's scan order


class PlayerHistory:
    def __init__(self, player: int, color: str, episode: str = ""):
        self.player, self.color = player, color
        self._handles: dict[int, int] = {}  # engine object ID -> handle; private routing table
        self.visible: dict[int, dict] = {}  # handle -> current visible fact
        self.tiles: dict[tuple[int, int, int], dict] = {}  # visible tile -> v1 tile record
        self.own: dict | None = None  # the player's own state as last reported
        self.last_seen: dict[int, dict] = {}  # handle -> last observed fact and when
        self._observed_at = {"day": None, "sequence": None}  # the latest capture or decision this player saw
        self.records: list[dict] = []
        self.turns_completed = 0
        self.battles_started = 0
        self.battles_completed = 0
        self.day = 0
        self.map_size: list[int] | None = None
        self.v1 = HistoryV1(episode, player)
        self._v1_objects: dict[int, dict] = {}  # handle -> v1 object record
        self._pending_action_id: int | None = None  # the submitted action awaiting its result
        self.actions_without_result = 0  # consumed by the engine, then superseded by the terminal
        self.actions_unconsumed = 0  # submitted, but the game ended before the engine took them

    # -- handles -----------------------------------------------------------
    def _observe(self, engine_id: int) -> int:
        """Handle for an object this player is observing now; assigned on first sight."""
        if engine_id not in self._handles:
            self._handles[engine_id] = len(self._handles) + 1
        return self._handles[engine_id]

    def _known(self, engine_id: int) -> int:
        if engine_id not in self._handles:
            raise ProtocolError("Reference to an object this player never observed")
        return self._handles[engine_id]

    def handle_count(self) -> int:
        return len(self._handles)

    # -- ingestion ---------------------------------------------------------
    def start(self, map_size: list[int]) -> None:
        """Public map dimensions from the bridge handshake."""
        if (not isinstance(map_size, list) or len(map_size) != 3
                or not all(type(x) is int and x > 0 for x in map_size)):
            raise ProtocolError("Bridge handshake must state the public map size")
        self.map_size = list(map_size)

    def ingest(self, message: dict) -> dict:
        """Fold one validated engine message and return its stored learner record (do not mutate)."""
        kind = message["type"]
        record = {"type": kind, "player": self.player, "color": self.color, "index": len(self.records)}
        if kind == "terminal":
            # Reconcile before the terminal's own events: they follow the last action the engine consumed.
            consumed = None
            if self._pending_action_id is not None:
                consumed = message["last_action_id"] == self._pending_action_id
                if consumed:
                    self.actions_without_result += 1
                else:
                    self.actions_unconsumed += 1
                self._pending_action_id = None
            self.v1.end(consumed)
        if kind in {"observation", "action_result", "decision", "terminal"}:
            record["events"] = [self._event(event) for event in message["events"]]
        if kind == "observation":
            record.update(self._observation(message))
        elif kind == "action_result":
            record.update(action_id=message["action_id"], action=message["action"], accepted=message["accepted"],
                          before=self._state(message["before"]), after=self._state(message["after"]))
            self.v1.result(message["action"], message["accepted"], self.day)
            self._pending_action_id = None
        elif kind == "decision":
            record.update(self._decision(message))
            self.v1.note_unrepresented("dialog")
        elif kind == "turn_end":
            self.turns_completed += 1
        elif kind == "terminal":
            record.update(outcome=message["outcome"], last_action_id=message["last_action_id"],
                          state=self._state(message["state"]))
        record.update(day=self.day, turn=self.turns_completed + 1,
                      battles_started=self.battles_started, battles_completed=self.battles_completed)
        self.records.append(record)
        return record

    def act(self, action: int, action_id: int) -> None:
        """The player's own adventure action, as delivered to the engine; the terminal may still void it."""
        self._pending_action_id = action_id
        self.v1.action(action, self.day)

    def _fact(self, entry: dict) -> dict:
        return {key: entry[key] for key in FACT_FIELDS}

    def _event(self, event: dict) -> dict:
        kind = event["kind"]
        if kind == "visible_update":
            return self._visible_update(event)
        if kind == "battle_start":
            self.battles_started += 1
            self.v1.note_unrepresented(kind)
            return {"kind": kind, "side": event.get("side"), "opponent": event.get("opponent")}
        if kind == "battle_complete":
            self.battles_completed += 1
            self.v1.note_unrepresented(kind)
            return {"kind": kind, "won": event.get("won")}
        if kind == "visit":
            # Delivered for this player's own hero interacting with the object,
            # which is itself a legitimate observation of it.
            self.v1.note_unrepresented(kind)
            return {"kind": kind, "handle": self._observe(event["object_id"]), "type": event.get("object_type"),
                    "subtype": event.get("object_subtype"), "owner": event.get("owner")}
        if kind == "level_up":
            # Applied by the server without a query: at most one skill was offered.
            self.v1.note_unrepresented(kind)
            return {"kind": kind, "primary_skill": event.get("primary_skill"),
                    "secondary_skills": list(event.get("secondary_skills", []))}
        if kind == "decision":
            self.v1.note_unrepresented("dialog_answer")
            return {"kind": kind, "decision_kind": event.get("decision_kind"),
                    "decision_id": event.get("decision_id"), "choice": event.get("choice")}
        raise ProtocolError(f"Unknown event kind: {kind}")

    def _visible_update(self, event: dict) -> dict:
        day = event.get("day")
        if type(day) is int:
            self.day = max(self.day, day)
        result = {"kind": "visible_update", "sequence": event["sequence"], "day": day, "appeared": [], "changed": [],
                  "disappeared": [], "revealed": [], "tiles_changed": [], "hidden": [], "own": None}
        for tile in event["hidden"]:
            key = tuple(tile)
            if key not in self.tiles:
                raise ProtocolError("Hidden tile was not visible")
            del self.tiles[key]
            result["hidden"].append(list(key))
        for name in ("revealed", "tiles_changed"):
            for tile in event[name]:
                key = (tile["x"], tile["y"], tile["z"])
                if (key in self.tiles) != (name == "tiles_changed"):
                    raise ProtocolError("Revealed tile was already visible" if name == "revealed"
                                        else "Tile facts changed for a tile the player does not see")
                self.tiles[key] = {"position": list(key), **{fact: tile[fact] for fact in TILE_FACTS}}
                result[name].append({"x": key[0], "y": key[1], "z": key[2], **{fact: tile[fact] for fact in TILE_FACTS}})
        for engine_id in event["disappeared"]:
            handle = self._known(engine_id)
            if handle not in self.visible:
                raise ProtocolError("Disappearing object was not visible")
            del self.visible[handle]
            del self._v1_objects[handle]
            # Keep the last observed position and attributes. It was last seen at the
            # previous capture or decision, which never precedes its own appearance.
            self.last_seen[handle].update(self._observed_at, visible=False)
            result["disappeared"].append(handle)
        # Handles follow visible facts, never the engine's object-ID order.
        appeared = sorted(event["appeared"], key=lambda e: (e["z"], e["y"], e["x"], e["type"], e["subtype"], e["owner"], e["id"]))
        for key, entries in (("appeared", appeared), ("changed", event["changed"])):
            for entry in entries:
                fact = self._fact(entry)
                position = (fact["x"], fact["y"], fact["z"])
                if position not in self.tiles:
                    raise ProtocolError("Visible object is reported on a tile the player does not see")
                if key == "appeared":
                    handle = self._observe(entry["id"])
                    if handle in self.visible:
                        raise ProtocolError("Appearing object was already visible")
                else:
                    handle = self._known(entry["id"])
                    if handle not in self.visible:
                        raise ProtocolError("Changed object was not visible")
                self.visible[handle] = fact
                self._v1_objects[handle] = {"handle": str(handle), "position": list(position), "kind": fact["type"],
                                            "subtype": fact["subtype"],
                                            "owner": relative_owner(fact["owner"], self.player)}
                self.last_seen[handle] = {"fact": dict(fact), "day": day, "sequence": event["sequence"], "visible": True}
                result[key].append({"handle": handle, **fact})
        for key in ("changed", "disappeared"):
            result[key].sort(key=lambda entry: entry if type(entry) is int else entry["handle"])
        if event["own"] is not None:
            self.own = copy.deepcopy(event["own"])
            result["own"] = copy.deepcopy(event["own"])
        if self.own is None:
            raise ProtocolError("The first visible update must carry the player's own state")
        self._observed_at = {"day": day, "sequence": event["sequence"]}
        self.v1.view(self.view(), self.day)
        return result

    def view(self) -> dict:
        """The current player-filtered view in v1 form; substructures are shared."""
        if self.map_size is None:
            raise ProtocolError("Visible updates arrived before the bridge handshake")
        return {"size": self.map_size, "tiles": [self.tiles[key] for key in sorted(self.tiles, key=_position_key)],
                "objects": [self._v1_objects[handle] for handle in sorted(self._v1_objects)], "own": self.own}

    def _state(self, state: dict) -> dict:
        result = {key: copy.deepcopy(value) for key, value in state.items() if key != "hero_id"}
        hero = state.get("hero_id", -1)
        result["hero"] = None if hero in (-1, None) else self._observe(hero)  # an own hero is always visible to its owner
        if type(state.get("day")) is int:
            self.day = max(self.day, state["day"])
        return result

    @staticmethod
    def _own_from_state(state: dict) -> dict:
        """The own-state fields of a decision snapshot, for the fold check."""
        hero = state.get("hero_id", -1)
        return {"hero": None if hero in (-1, None) else [state.get("x"), state.get("y"), state.get("z")],
                "movement": state.get("movement"), "movement_max": state.get("movement_max"),
                "resources": state.get("resources"), "army": state.get("army", []), "towns": state.get("towns"),
                "mines": state.get("own_mines")}

    def _observation(self, message: dict) -> dict:
        self.day = max(self.day, message["day"])
        snapshot_tiles, snapshot_objects, tiles = {}, {}, []
        snapshot = "tiles" in message  # the engine may send the redundant tile list periodically
        for tile in message.get("tiles", []):
            key = (tile["x"], tile["y"], tile["z"])
            snapshot_tiles[key] = {"position": list(key), **{fact: tile.get(fact) for fact in TILE_FACTS}}
            objects = []
            for entry in tile.get("objects", []):
                engine_id = entry["id"]
                if engine_id not in self._handles:
                    raise ProtocolError("Decision snapshot shows an object absent from the visible event history")
                handle = self._handles[engine_id]
                fact = {"x": key[0], "y": key[1], "z": key[2], "type": entry["type"],
                        "subtype": entry["subtype"], "owner": entry["owner"]}
                snapshot_objects.setdefault(handle, fact)  # scan order matches the native tracker
                objects.append({"handle": handle, "type": entry["type"], "subtype": entry["subtype"], "owner": entry["owner"]})
            tiles.append({"x": key[0], "y": key[1], "z": key[2], **{fact: tile.get(fact) for fact in TILE_FACTS},
                          "objects": objects})
        if snapshot and (snapshot_tiles != self.tiles or snapshot_objects != self.visible):
            raise ProtocolError("Visible event history diverged from the decision snapshot")
        if self._own_from_state(message["state"]) != self.own:
            raise ProtocolError("Visible event history diverged from the decision snapshot")
        if [message.get("width"), message.get("height"), message.get("levels")] != self.map_size:
            raise ProtocolError("Decision snapshot map size differs from the handshake")
        self.v1.decision_frame(self.view(), message["action_mask"], self.day)
        # A decision shows the whole current view: everything visible is seen again now.
        self._observed_at = {**self._observed_at, "day": message["day"]}
        hero = message.get("hero_id", -1)
        return {"action_id": message["action_id"], "action_mask": list(message["action_mask"]),
                "hero": None if hero in (-1, None) else self._observe(hero),
                "x": message.get("x"), "y": message.get("y"), "z": message.get("z"),
                "width": message.get("width"), "height": message.get("height"), "levels": message.get("levels"),
                "movement": message.get("movement"), "movement_max": message.get("movement_max"),
                "tiles": tiles if snapshot else None, "state": self._state(message["state"])}

    def _decision(self, message: dict) -> dict:
        # An allowlist: a new engine field must be reviewed before it can reach a learner.
        unknown = set(message) - DECISION_FIELDS - {"type", "player", "episode", "protocol_version", "events", "hero_id"}
        if unknown:
            raise ProtocolError(f"Decision carries unreviewed fields: {sorted(unknown)}")
        result = {key: copy.deepcopy(message[key]) for key in DECISION_FIELDS if key in message and key != "choices"}
        result["choices"] = []
        for choice in message["choices"]:
            if not isinstance(choice, dict) or set(choice) - CHOICE_FIELDS:
                raise ProtocolError("Decision choice carries unreviewed fields")
            result["choices"].append(dict(choice))
        if "hero_id" in message:
            result["hero"] = self._observe(message["hero_id"])
        return result

    def memory(self) -> list[dict]:
        """Last-seen facts in handle order; ``visible`` separates history from the present.

        A visible object was last seen at the latest capture or decision; a
        hidden one keeps the time of its last observation.
        """
        return [{"handle": handle, **copy.deepcopy(entry), **(self._observed_at if entry["visible"] else {})}
                for handle, entry in sorted(self.last_seen.items())]
