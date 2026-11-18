"""Wire protocol between the engine's AI plugin and one seat's typed history.

Each seat has its own socket and its own ``Producer``. The engine sends
newline-delimited JSON: ``ready`` once, then batches of records. The producer
turns every record into a validated ``contract.Frame``, assigns the frame
sequence numbers and derives the legal actions from the observation itself;
the engine sends no action mask.
"""
from __future__ import annotations

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


def unique_object(pairs):
    # dict() keeps first insertion order and drops nothing unless a key repeats;
    # only then does the loop name the first repeated key, as it always did.
    result = dict(pairs)
    if len(result) == len(pairs):
        return result
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def loads(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid)


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
        self.terminal = None  # this seat's "win" or "loss" once the engine sent it
        self.decision_ids = {}  # engine decision id -> frame sequence number
        self.current_id = None

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
            exact(record, {"kind", "day"} | RECORD_FIELDS[record_kind], record_kind)
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
            observation = None
            if "observation" in raw:
                if declared_army_information(raw["observation"], self.profile.environment) != self.army_information:
                    raise ValueError("Observation army information differs from the negotiated extension")
            if record_kind == "decision":
                observation = contract.decode_observation(raw["observation"], schema=self.profile.history_schema)
                raw["offers"] = contract.encode_offers(contract.Offers.from_observation(observation, schema=self.profile.action_schema))
            # The decision's observation is decoded once, from this same raw dict and schema.
            frame = contract.decode_frame(raw, observation=observation)
            self.history.append(frame)
            appended.append(frame)
            if record_kind == "decision":
                self.decision_ids[native_id] = frame.seq
                self.current_id = native_id
            elif record_kind in ("result", "superseded"):
                self.current_id = None
            elif record_kind == "terminal":
                self.terminal = frame.outcome
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
