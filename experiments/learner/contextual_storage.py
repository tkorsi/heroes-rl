"""Lossless, manifest-bound rollout storage without a second copy of history.

The envelope is a seal, written last and never overwritten. Diagnostics retain
their original player JSONL files. Fresh owned capture-only training uses a
restricted binary handoff bound to child completion. Both retain exact decisions,
supports, probabilities, values and sample boundaries. Neither codec grants
verification or PPO admission. Legacy self-contained artifacts remain readable.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import io
import pickle
import json
import os
from pathlib import Path
import tempfile

from experiments.headless import contextual_evidence as evidence
from . import contextual as c
from .contextual_rollout import (CollectedEpisode, ContextualRollout, _TRAJECTORIES,
                                 _canonical, _record)

SCHEMA = "heroes-context-rollout-storage-v1"
LIVE_SCHEMA = "heroes-owned-context-rollout-transfer-v1"
LIVE_FILE = "training-rollout.bin"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata(rollout):
    result = {item.name: getattr(rollout, item.name) for item in fields(rollout)
              if item.init and item.name not in ("history", "decisions", "samples", "estimates")}
    if rollout.trajectory_schema == c.TRAJECTORY_SCHEMA:
        result.pop("trajectory_schema")
    result.update(decisions=[item.to_dict() for item in rollout.decisions],
                  samples=[item.to_dict(schema=rollout.trajectory_schema) for item in rollout.samples],
                  estimates=[item.to_dict() for item in rollout.estimates])
    return result


def _reference(folder, manifest, player, expected=None):
    profile = c.profile_for(manifest["environment"])
    declaration = manifest["players"][("red", "blue")[player]][profile.history_key]
    name = declaration["path"]
    path = evidence.inside(folder, name)
    identity = evidence.file_identity(path)
    _require(identity["sha256"] == declaration["sha256"], "Stored history differs from its manifest")
    reference = dict(path=name, **identity, canonical_sha256=declaration["canonical_sha256"])
    if expected is not None:
        _require(reference == expected, "External history reference changed")
    return reference


def save_collected(path, collected):
    """Seal a fresh artifact referencing the already finalized player histories."""
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), "Refusing to replace an existing rollout")
    manifest = evidence.strict_json(evidence.inside(path.parent, "manifest.json").read_bytes())
    _require(manifest == json.loads(collected.manifest_json), "Persisted rollout manifest differs")
    rollouts = []
    for player, rollout in enumerate(collected.rollouts):
        reference = _reference(path.parent, manifest, player)
        _require(rollout.player == player and reference["canonical_sha256"] == rollout.history_sha256,
                 "External history does not bind this player rollout")
        rollouts.append(dict(history_ref=reference, records=_metadata(rollout)))
    envelope = dict(schema=SCHEMA, artifact_schema=collected.artifact_schema, rollouts=rollouts,
                    manifest=manifest, audit=json.loads(collected.audit_json),
                    model_configs=[json.loads(value) for value in collected.model_configs_json])
    _seal(path, envelope)


def _seal(path, envelope):
    # fsync the complete temporary file before an exclusive atomic publication.
    # A crash leaves either no seal or a complete seal, never a partial success.
    fd, temporary = tempfile.mkstemp(prefix=".rollout-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(envelope, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if another writer already published the name.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)



class _OwnedTransferReader(pickle.Unpickler):
    """Only the immutable records our owned collector writes, never executable globals."""

    def find_class(self, module, name):
        allowed = ('experiments.learner.contextual', 'experiments.learner.contextual_model',
                   'experiments.learner.contextual_rollout', 'experiments.learner.interaction_guard')
        if module in allowed:
            cls = getattr(importlib.import_module(module), name, None)
            if isinstance(cls, type) and is_dataclass(cls) and cls.__module__ == module:
                return cls
        raise ValueError('Unsupported class in owned training transfer')


def save_live_collected(path, collected):
    """Seal a validated in-process capture for its owning coordinator's handoff."""
    path = Path(path)
    _require(type(collected) is CollectedEpisode and all(
        r.origin == 'native' and r.purpose == 'training' for r in collected.rollouts),
        'Binary transfer is only for fresh native training')
    manifest = evidence.strict_json((path.parent / 'manifest.json').read_bytes())
    _require(manifest.get('capture_only') is True and manifest == json.loads(collected.manifest_json),
             'Binary transfer requires its explicit capture-only manifest')
    _require(not path.exists() and not path.is_symlink(), 'Refusing to replace an existing transfer')
    blob = path.parent / LIVE_FILE
    with blob.open('xb') as stream:
        pickle.dump(collected, stream, protocol=5)
        stream.flush(); os.fsync(stream.fileno())
    _seal(path, dict(schema=LIVE_SCHEMA, manifest=manifest,
                    transfer=dict(path=LIVE_FILE, **evidence.file_identity(blob))))


def load_live_collected(path, expected_files):
    """A completion-token owner must bind all bytes before using this fast reader.

    This reader supplies no verification or admission capability. The general
    legacy/diagnostic loader intentionally does not accept this private format.
    """
    path = Path(path)
    envelope_bytes = path.read_bytes()
    _require(evidence.file_identity(path) == expected_files['neural-rollout.json'], 'Transfer seal changed')
    envelope = _record(evidence.strict_json(envelope_bytes), ('schema', 'manifest', 'transfer'))
    _require(envelope['schema'] == LIVE_SCHEMA, 'Unknown owned training transfer')
    reference = _record(envelope['transfer'], ('path', 'sha256', 'bytes'))
    _require(reference['path'] == LIVE_FILE
             and {k:v for k,v in reference.items() if k != 'path'} == expected_files[LIVE_FILE],
             'Owned transfer differs from its actual child completion')
    payload = evidence.inside(path.parent, LIVE_FILE).read_bytes()
    _require(len(payload) == reference['bytes'] and hashlib.sha256(payload).hexdigest() == reference['sha256'],
             'Owned binary transfer changed')
    stream = io.BytesIO(payload)
    collected = _OwnedTransferReader(stream).load()
    _require(stream.read(1) == b'' and type(collected) is CollectedEpisode,
             'Expected one complete owned rollout transfer')
    manifest = evidence.strict_json((path.parent / 'manifest.json').read_bytes())
    _require(evidence.file_identity(path.parent / 'manifest.json') == expected_files['manifest.json']
             and manifest == envelope['manifest'] == json.loads(collected.manifest_json),
             'Owned transfer manifest changed')
    return collected


def _history(folder, reference):
    reference = _record(reference, ("path", "sha256", "bytes", "canonical_sha256"))
    _require(type(reference["bytes"]) is int and reference["bytes"] > 0, "Invalid history byte count")
    path = evidence.inside(folder, reference["path"])
    digest, size, frames = hashlib.sha256(), 0, []
    with path.open("rb") as stream:
        for line in stream:
            _require(line.endswith(b"\n"), "Incomplete external history record")
            digest.update(line); size += len(line)
            row = evidence.strict_json(line)
            frame = c.decode_frame(row)
            encoded = c.encode_frame(frame)
            _require(encoded == row, "External history record does not roundtrip exactly")
            # Reuse only the just-validated canonical representation of this
            # immutable frame; no file cache or persisted proof is trusted.
            object.__setattr__(frame, "_canonical_json", _canonical(encoded))
            frames.append(frame)
    _require(size == reference["bytes"] and digest.hexdigest() == reference["sha256"],
             "External history bytes changed while loading")
    return tuple(frames)


def load_collected(path):
    """Strictly reload every record; no file cache or persisted verification trust."""
    path = Path(path)
    raw = evidence.strict_json(path.read_bytes())
    if raw.get("schema") == LIVE_SCHEMA:
        raise ValueError("Owned training transfer requires its fresh child completion binding")
    if raw.get("schema") != SCHEMA:
        collected = CollectedEpisode.from_dict(raw)
        _require(collected.to_dict() == raw, "Legacy rollout does not roundtrip exactly")
        return collected
    data = _record(raw, ("schema", "artifact_schema", "rollouts", "manifest", "audit", "model_configs"))
    _require(data["artifact_schema"] in _TRAJECTORIES and type(data["rollouts"]) is list
             and len(data["rollouts"]) == 2 and type(data["model_configs"]) is list,
             "Invalid external rollout envelope")
    manifest = evidence.strict_json(evidence.inside(path.parent, "manifest.json").read_bytes())
    _require(data["manifest"] == manifest, "External rollout manifest differs from its file")
    rollouts, histories = [], []
    try:
        for player, entry in enumerate(data["rollouts"]):
            entry = _record(entry, ("history_ref", "records"))
            reference = _reference(path.parent, manifest, player, entry["history_ref"])
            history = _history(path.parent, reference)
            histories.append(history)
            rollout = ContextualRollout.from_decoded_history(entry["records"], history,
                trajectory_schema=_TRAJECTORIES[data["artifact_schema"]])
            _require(rollout.player == player and rollout.history_sha256 == reference["canonical_sha256"]
                     and _metadata(rollout) == entry["records"], "External rollout records changed on reconstruction")
            rollouts.append(rollout)
        return CollectedEpisode(tuple(rollouts), _canonical(manifest), _canonical(data["audit"]),
                                tuple(_canonical(value) for value in data["model_configs"]), data["artifact_schema"])
    finally:
        # Paired validation uses these same immutable frames. Release after
        # that final check, instead of serializing both histories again.
        for history in histories:
            c.release_canonical(history)
