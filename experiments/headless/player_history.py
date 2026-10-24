"""One player's sanitized, ordered learner history.

The engine stream carries engine object IDs only so that this adapter can route
facts. They never leave it: each player gets opaque, episode-local handles,
assigned in order of that player's own first legitimate observation. Handles
are never created for an object that this player has not seen, so a handle
cannot leak the existence of a hidden object. Two players' histories share no
state; the same engine object normally has different handles for each.

Visible updates are folded into the player's current view and a last-seen
record. A disappearance only marks a remembered fact as no longer visible: its
position and attributes stay those of the last observation and never refresh
from hidden state. At every decision snapshot the folded view must equal the
snapshot's visible tiles and objects, which proves that the event stream alone
reconstructs what this player saw between decisions.
"""
from __future__ import annotations

import copy

from bridge_protocol import ProtocolError

FACT_FIELDS = ("x", "y", "z", "type", "subtype", "owner")


class PlayerHistory:
    def __init__(self, player: int, color: str):
        self.player, self.color = player, color
        self._handles: dict[int, int] = {}  # engine object ID -> handle; private routing table
        self.visible: dict[int, dict] = {}  # handle -> current visible fact
        self.tiles: set[tuple[int, int, int]] = set()
        self.last_seen: dict[int, dict] = {}  # handle -> last observed fact and when
        self.records: list[dict] = []
        self.turns_completed = 0
        self.battles_started = 0
        self.battles_completed = 0
        self.day = 0

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
    def ingest(self, message: dict, step: int | None = None) -> dict:
        """Fold one validated engine message and return its learner record."""
        kind = message["type"]
        record = {"type": kind, "player": self.player, "color": self.color, "index": len(self.records)}
        if step is not None:
            record["env_step"] = step
        if kind in {"observation", "action_result", "decision", "terminal"}:
            record["events"] = [self._event(event) for event in message["events"]]
        if kind == "observation":
            record.update(self._observation(message))
        elif kind == "action_result":
            record.update(action_id=message["action_id"], action=message["action"], accepted=message["accepted"],
                          before=self._state(message["before"]), after=self._state(message["after"]))
        elif kind == "decision":
            record.update(self._decision(message))
        elif kind == "turn_end":
            self.turns_completed += 1
        elif kind == "terminal":
            record.update(outcome=message["outcome"], state=self._state(message["state"]))
        record.update(day=self.day, turn=self.turns_completed + 1,
                      battles_started=self.battles_started, battles_completed=self.battles_completed)
        self.records.append(record)
        return copy.deepcopy(record)

    def _fact(self, entry: dict) -> dict:
        return {key: entry[key] for key in FACT_FIELDS}

    def _event(self, event: dict) -> dict:
        kind = event["kind"]
        if kind == "visible_update":
            return self._visible_update(event)
        if kind == "battle_start":
            self.battles_started += 1
            return {"kind": kind, "side": event.get("side"), "opponent": event.get("opponent")}
        if kind == "battle_complete":
            self.battles_completed += 1
            return {"kind": kind, "won": event.get("won")}
        if kind == "visit":
            # Delivered for this player's own hero interacting with the object,
            # which is itself a legitimate observation of it.
            return {"kind": kind, "handle": self._observe(event["object_id"]), "type": event.get("object_type"),
                    "subtype": event.get("object_subtype"), "owner": event.get("owner")}
        if kind == "decision":
            return {"kind": kind, "decision_kind": event.get("decision_kind"),
                    "decision_id": event.get("decision_id"), "choice": event.get("choice")}
        raise ProtocolError(f"Unknown event kind: {kind}")

    def _visible_update(self, event: dict) -> dict:
        day = event.get("day")
        if type(day) is int:
            self.day = max(self.day, day)
        result = {"kind": "visible_update", "sequence": event["sequence"], "day": day,
                  "appeared": [], "changed": [], "disappeared": [], "revealed": [], "hidden": []}
        for tile in event["revealed"]:
            key = tuple(tile)
            if key in self.tiles:
                raise ProtocolError("Revealed tile was already visible")
            self.tiles.add(key)
            result["revealed"].append(list(key))
        for tile in event["hidden"]:
            key = tuple(tile)
            if key not in self.tiles:
                raise ProtocolError("Hidden tile was not visible")
            self.tiles.discard(key)
            result["hidden"].append(list(key))
        for engine_id in event["disappeared"]:
            handle = self._known(engine_id)
            if handle not in self.visible:
                raise ProtocolError("Disappearing object was not visible")
            del self.visible[handle]
            # Keep the last observed position and attributes; only visibility changes.
            self.last_seen[handle]["visible"] = False
            result["disappeared"].append(handle)
        for key in ("appeared", "changed"):
            for entry in event[key]:
                fact = self._fact(entry)
                if (fact["x"], fact["y"], fact["z"]) not in self.tiles:
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
                self.last_seen[handle] = {"fact": dict(fact), "day": day, "sequence": event["sequence"], "visible": True}
                result[key].append({"handle": handle, **fact})
        return result

    def _state(self, state: dict) -> dict:
        result = {key: copy.deepcopy(value) for key, value in state.items() if key != "hero_id"}
        hero = state.get("hero_id", -1)
        result["hero"] = None if hero in (-1, None) else self._observe(hero)  # an own hero is always visible to its owner
        if type(state.get("day")) is int:
            self.day = max(self.day, state["day"])
        return result

    def _observation(self, message: dict) -> dict:
        self.day = max(self.day, message["day"])
        snapshot_tiles, snapshot_objects, tiles = set(), {}, []
        for tile in message["tiles"]:
            key = (tile["x"], tile["y"], tile["z"])
            snapshot_tiles.add(key)
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
            tiles.append({"x": key[0], "y": key[1], "z": key[2], "terrain": tile.get("terrain"),
                          "blocked": tile.get("blocked"), "visitable": tile.get("visitable"), "objects": objects})
        if snapshot_tiles != self.tiles or snapshot_objects != self.visible:
            raise ProtocolError("Visible event history diverged from the decision snapshot")
        hero = message.get("hero_id", -1)
        return {"action_id": message["action_id"], "action_mask": list(message["action_mask"]),
                "hero": None if hero in (-1, None) else self._observe(hero),
                "x": message.get("x"), "y": message.get("y"), "z": message.get("z"),
                "width": message.get("width"), "height": message.get("height"), "levels": message.get("levels"),
                "movement": message.get("movement"), "movement_max": message.get("movement_max"),
                "tiles": tiles, "state": self._state(message["state"])}

    def _decision(self, message: dict) -> dict:
        result = {key: copy.deepcopy(value) for key, value in message.items()
                  if key not in {"type", "player", "episode", "protocol_version", "events", "hero_id"}}
        if "hero_id" in message:
            result["hero"] = self._observe(message["hero_id"])
        return result

    def memory(self) -> list[dict]:
        """Last-seen facts in handle order; ``visible`` separates history from the present."""
        return [{"handle": handle, **copy.deepcopy(entry)} for handle, entry in sorted(self.last_seen.items())]
