"""Strict versioned transport to immutable player-local contextual history.

Raw transport identifiers are evidence only. Python assigns learner sequence
numbers and derives support solely from the delivered observation. Socket send,
consumer acknowledgement, native issuance and native acceptance stay separate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import json

from experiments.learner import contextual as contract

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

PROTOCOL = 4

# No caller-supplied offers, sequence numbers, submitted actions or final proofs.
RECORD_FIELDS = {
    "decision": {"decision_id", "revision", "observation"},
    "view": {"observation"}, "event": {"event"},
    "consumed": {"decision_id", "consumer"},
    "issued": {"decision_id", "command_index"},
    "acknowledged": {"decision_id", "command_index", "accepted"},
    "result": {"decision_id", "result"},
    "context": {"decision_id", "context", "revision"},
    "context_invalidated": {"context", "revision", "reason"},
    "superseded": {"decision_id", "reason"},
    "terminal": {"outcome"},
}


def unique_object(pairs):
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


def clone_history(history):
    cloned = contract.PlayerHistory(history.episode, history.player, schema=history.schema)
    for frame in history.frames:
        cloned.append(frame)
    return cloned


class ContextualProducer:
    """One socket recipient. No other player's state is available here."""

    def __init__(self, episode, player, *, interface=contract.ENVIRONMENT):
        self.episode, self.player = episode, player
        self.profile = contract.profile_for(interface)
        if interface != self.profile.environment:
            raise ValueError("Producer requires an explicit environment interface")
        self.live = contract.PlayerHistory(episode, player, schema=self.profile.history_schema)
        self.history = self.live
        self.ready = False
        self.terminal = None
        self.decision_ids = {}
        self.current_id = None
        self.cutoff = None
        self.continuation_steps = 0

    def ingest(self, message):
        if type(message) is not dict or message.get("protocol_version") != self.profile.protocol or type(message.get("protocol_version")) is not int:
            raise ValueError(f"Contextual mode requires explicit protocol_version {self.profile.protocol}")
        kind = message.get("type")
        base = {"protocol_version", "type", "episode", "player"}
        if message.get("episode") != self.episode or type(message.get("player")) is not int or message.get("player") != self.player:
            raise ValueError("Cross-player or cross-episode contextual envelope")
        if kind == "ready":
            exact(message, base | {"interface", "history_schema", "action_schema"}, "ready")
            if (self.ready or message["interface"] != self.profile.environment or message["history_schema"] != self.profile.history_schema
                    or message["action_schema"] != self.profile.action_schema):
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
            raw.update(schema=self.profile.history_schema, episode=self.episode, player=self.player, seq=len(self.live.frames))
            if record_kind == "decision":
                observation = contract.decode_observation(raw["observation"], schema=self.profile.history_schema)
                raw["offers"] = contract.encode_offers(contract.Offers.from_observation(observation, schema=self.profile.action_schema))
            frame = contract.decode_frame(raw)
            self.live.append(frame)
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
        decision = self.live.decision
        if (type(action) is not contract.Action or decision is None or self.live.action is not None
                or self.current_id is None or not decision.offers.permits(action)):
            raise ValueError("Action does not match the current contextual request")
        return {"protocol_version": self.profile.protocol, "type": "context_action", "episode": self.episode,
                "player": self.player, "decision_id": self.current_id, "revision": decision.revision,
                "action": contract.encode_action(action)}

    def submitted(self, action):
        # Called only after sendall succeeds; this is not proof of native receipt.
        self.response(action)
        frame = self.live.emit("submitted", action=action, decision_seq=self.live.decision.seq)
        if self.cutoff is not None:
            self.continuation_steps += 1
        return frame

    def freeze_cutoff(self, reason):
        if self.cutoff is not None or self.live.decision is None or self.live.action is not None:
            raise ValueError("Cutoff requires an actual unacted own decision")
        self.history = clone_history(self.live)
        self.cutoff = self.live.decision
        self.history.emit("cutoff", decision_seq=self.cutoff.seq, reason=reason)

    def fail(self, reason):
        if self.history.boundary not in (None, "cutoff_pending"):
            raise ValueError("Cannot invalidate an already finalized history")
        self.history.emit("failure", reason=reason)


class NativeFailure(ValueError):
    """An explicit engine unsupported/error envelope, never a game result."""
