"""Wire protocol between the engine's AI plugin and one seat's typed history.

Each seat has its own socket and its own ``Producer``. The engine sends
newline-delimited JSON: ``ready`` once, then batches of records. The producer
turns every record into a validated ``contract.Frame``, assigns the frame
sequence numbers and derives the legal actions from the observation itself;
the engine sends no action mask.

A decision is always about the observation of the seat's latest ``view``
record. An engine started with ``EXPLORER_RL_DECISION_OBSERVATION=0`` therefore
sends the decision without it, and the decision frame gets the view's
observation, the same object. An engine that still repeats the observation is
read as before; the frames are equal either way.

Most views change a window and leave the map as it was. An engine started with
``EXPLORER_RL_REPEATED_MAP=0`` sends such a view without ``tiles`` and
``objects``, and the view's observation gets those of the seat's previous view,
the same tuples. With ``EXPLORER_RL_TILE_CHANGES=1`` a view that changed the map
while the same tiles stayed visible sends its objects and, as ``tile_changes``,
only the tiles that differ; its observation gets the previous tiles with those
replaced. An engine that sends the whole map every time is read as before; the
frames are equal either way.
"""
from __future__ import annotations

import marshal

import json

from ..game import contract

# Payload fields of each record kind, besides ``kind`` and ``day``.
RECORD_FIELDS = {
    "decision": {"decision_id", "revision", "observation"},
    "view": {"observation"}, "event": {"event"},
    "consumed": {"decision_id", "consumer"},
    "issued": {"decision_id", "command_index"},
    "acknowledged": {"decision_id", "command_index", "accepted"},
    "result": {"decision_id", "result"},
    "context": {"decision_id", "context", "revision"},
    "context_invalidated": {"context", "revision", "reason"},
    "context_opened": {"context", "revision", "reason"},
    "superseded": {"decision_id", "reason"},
    "terminal": {"outcome"},
}


def loads(data):
    """One engine message. The engine writes every object from a map, so no key repeats."""
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(data, parse_constant=invalid)


def exact(raw, fields, name):
    if type(raw) is not dict or set(raw) != fields:
        raise ValueError(f"Unexpected or missing {name} fields")


def validate_army_information(interface, value):
    if value is not None and (interface != contract.EXPANDED_ENVIRONMENT or value != contract.OWN_ARMY_INFORMATION):
        raise ValueError("Own-army information requires the explicit supported v8 extension")
    return value


def declared_army_information(document, interface):
    if type(document) is not dict:
        raise ValueError("Army information declaration requires an object")
    value = document.get("army_information")
    if "army_information" in document and value is None:
        raise ValueError("Absent army information must omit its declaration")
    return validate_army_information(interface, value)


_VIEWS = {}  # (marshal bytes of a raw observation, ids of the map it was read against) -> (observation, that map)


def _decoded_view(raw, schema, unchanged_map):
    """``contract.decode_observation`` of a view, as one shared object for a raw observation seen before.

    Half of a seat's views repeat an earlier one exactly: the same window
    opened again. Decoding is a function of the raw value, the schema and, for a
    view that leaves out its map or lists only changed tiles, the map of the
    view before; equal marshal bytes mean equal values of identical exact types,
    and the entry holds the map it was read against, so its ids name nothing
    else. Only a successful decoding is kept.
    """
    try:
        data = marshal.dumps(raw)
    except ValueError:
        return contract.decode_observation(raw, schema=schema, unchanged_map=unchanged_map)
    reads_map = type(raw) is dict and "tiles" not in raw
    key = (data, schema, *(map(id, unchanged_map) if reads_map and unchanged_map is not None else ()))
    known = _VIEWS.get(key)
    if known is not None and (not reads_map or unchanged_map is None or all(
            held is value for held, value in zip(known[1], unchanged_map))):
        return known[0]
    observation = contract.decode_observation(raw, schema=schema, unchanged_map=unchanged_map)
    if len(_VIEWS) >= 2048:
        _VIEWS.clear()  # bounded memory; a miss only decodes again
    _VIEWS[key] = (observation, unchanged_map if reads_map else None)
    return observation


# Every field a record of each kind carries, "kind" and "day" included; True: a decision without its observation.
_EXACT_FIELDS = {(kind, bare): frozenset({"kind", "day"} | (fields - {"observation"} if bare else fields))
                 for kind, fields in RECORD_FIELDS.items() for bare in (False, True)}


class Producer:
    """One seat's stream. No other player's state is available here."""

    def __init__(self, episode, player, *, interface=contract.ENVIRONMENT, army_information=None):
        self.episode, self.player = episode, player
        self.profile = contract.profile_for(interface)
        if interface != self.profile.environment:
            raise ValueError("Producer requires an explicit environment interface")
        self.army_information = validate_army_information(interface, army_information)
        self.history = contract.PlayerHistory(episode, player, schema=self.profile.history_schema)
        self.ready = False
        self.gold_mine_captures = 0
        self.terminal = None  # this seat's "win" or "loss" once the engine sent it
        self.decision_ids = {}  # engine decision id -> frame sequence number
        self.current_id = None
        self.view = None  # the observation of the latest view record

    def ingest(self, message):
        if type(message) is not dict or message.get("protocol_version") != self.profile.protocol or type(message.get("protocol_version")) is not int:
            raise ValueError(f"Contextual mode requires explicit protocol_version {self.profile.protocol}")
        kind = message.get("type")
        base = {"protocol_version", "type", "episode", "player"}
        if message.get("episode") != self.episode or type(message.get("player")) is not int or message.get("player") != self.player:
            raise ValueError("Cross-player or cross-episode contextual envelope")
        if kind == "ready":
            extension = {"army_information"} if self.army_information is not None else set()
            exact(message, base | {"interface", "history_schema", "action_schema"} | extension, "ready")
            if (self.ready or message["interface"] != self.profile.environment or message["history_schema"] != self.profile.history_schema
                    or message["action_schema"] != self.profile.action_schema
                    or declared_army_information(message, self.profile.environment) != self.army_information):
                raise ValueError("Duplicate ready or mismatched contextual schema")
            self.ready = True
            return ()
        if kind == "error":
            exact(message, base | {"message"}, "error")
            if type(message["message"]) is not str or not message["message"].strip():
                raise ValueError("Native error must identify the failure")
            raise NativeFailure(message["message"])
        exact(message, base | {"records"}, "contextual envelope")
        if kind != "contextual" or not self.ready or type(message["records"]) is not list or not 1 <= len(message["records"]) <= 100000:
            raise ValueError("Expected a nonempty contextual record batch after ready")
        appended = []
        for record in message["records"]:
            if type(record) is not dict or record.get("kind") not in RECORD_FIELDS:
                raise ValueError("Unsupported contextual record kind")
            record_kind = record["kind"]
            # The decision without an observation has the one of the latest view.
            exact(record, _EXACT_FIELDS[record_kind, record_kind == "decision" and "observation" not in record], record_kind)
            raw = dict(record)
            if "decision_id" in raw:
                native_id = raw.pop("decision_id")
                if type(native_id) is not int or native_id < 0:
                    raise ValueError("decision_id must be a nonnegative native-local integer")
                if record_kind == "decision":
                    if native_id in self.decision_ids:
                        raise ValueError("Repeated native decision_id")
                elif native_id not in self.decision_ids:
                    raise ValueError("Unknown native decision reference")
                else:
                    raw["decision_seq"] = self.decision_ids[native_id]
            raw.update(schema=self.profile.history_schema, episode=self.episode, player=self.player, seq=self.history.frame_count)
            observation = offers = None
            if "observation" in raw:
                if declared_army_information(raw["observation"], self.profile.environment) != self.army_information:
                    raise ValueError("Observation army information differs from the negotiated extension")
            if record_kind == "view":
                # A view without its map has the map of the view before it.
                observation = _decoded_view(raw["observation"], self.profile.history_schema,
                                            None if self.view is None else (self.view.tiles, self.view.objects))
            elif record_kind == "decision":
                if "observation" in raw:
                    observation = contract.decode_observation(raw["observation"], schema=self.profile.history_schema)
                elif self.view is None:
                    raise ValueError("A decision without an observation needs an earlier view")
                else:
                    observation = self.view
                # The offers are built from the observation and handed over as they are: encoding them to their
                # wire form and decoding that again gave an equal value, built twice.
                offers = contract.Offers.from_observation(observation, schema=self.profile.action_schema)
            # An observation is decoded once: here, or for a decision when its view arrived.
            frame = contract.decode_frame(raw, observation=observation, offers=offers)
            self.history.append(frame)
            appended.append(frame)
            if record_kind == "view":
                self.view = frame.observation
            elif record_kind == "decision":
                self.decision_ids[native_id] = frame.seq
                self.current_id = native_id
            elif record_kind in ("result", "superseded"):
                self.current_id = None
            elif record_kind == "terminal":
                self.terminal = frame.outcome
            elif record_kind == "event" and frame.event.kind == "notice":
                notice = frame.event.notice
                # CGMine::flagMine sends this notice to the player taking ownership.
                if (notice.display_mode == "auto" and len(notice.text) == 1
                        and notice.text[0].op == "append_text"
                        and notice.text[0].text_id == "core.mineevnt.6"
                        and len(notice.components) == 1
                        and notice.components[0].kind == "resource_per_day"
                        and notice.components[0].resource == "gold"):
                    self.gold_mine_captures += 1
        return tuple(appended)

    def response(self, action):
        decision = self.history.decision
        if (type(action) is not contract.Action or decision is None or self.history.action is not None
                or self.current_id is None or not decision.offers.permits(action)):
            raise ValueError("Action does not match the current contextual request")
        return {"protocol_version": self.profile.protocol, "type": "context_action", "episode": self.episode,
                "player": self.player, "decision_id": self.current_id, "revision": decision.revision,
                "action": contract.encode_action(action)}

    def submitted(self, action):
        """Record an answer after it was written to the socket."""
        self.response(action)
        return self.history.emit("submitted", action=action, decision_seq=self.history.decision.seq)


class NativeFailure(ValueError):
    """The engine reported an error or an unsupported situation; never a game result."""
