"""Reconstruct contextual histories from transport, without a game or model.

Saved learner/live histories are comparison targets only. The bridge supplies
native records and delivered intentions; the hash-bound manifest supplies the
cutoff location and paired finalization metadata. This verifies transport to
history equivalence, not native packet semantics or terminal-barrier validity.
Those independent checks belong to contextual_replay.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from experiments.episodes.scoring import HEADLESS_TRUNCATIONS
from experiments.headless import contextual_evidence as evidence
from experiments.headless.contextual_protocol import ContextualProducer, exact
from experiments.learner import contextual as contract

COLORS = ("red", "blue")
CONTINUATION_LIMIT = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _identity(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _wire(frame):
    return json.dumps(contract.encode_frame(frame), sort_keys=True,
                      separators=(",", ":")).encode() + b"\n"


def _records(path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            _require(line.endswith(b"\n"), f"Incomplete JSONL record {path}:{number}")
            row = evidence.strict_json(line)
            _require(type(row) is dict, f"Non-object JSONL record {path}:{number}")
            yield row


def _bound_metadata(directory, manifest, final):
    on_disk = evidence.strict_json(evidence.inside(directory, "manifest.json").read_bytes())
    _require(manifest == on_disk, "Supplied manifest differs from its file")
    _require(type(manifest) is dict and manifest.get("kind") == "contextual-headless-self-play"
             and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
             "Unsupported contextual manifest")
    profile = contract.profile_for(manifest.get("environment"))
    _require(manifest["environment"] == profile.environment
             and manifest.get("history_schema") == profile.history_schema
             and manifest.get("action_schema") == profile.action_schema,
             "Mismatched manifest interface/history/action versions")
    _require(type(manifest.get("episode")) is str and bool(manifest["episode"]), "Invalid episode")
    reference = manifest.get("evidence_ref")
    _require(type(reference) is str, "Missing finalization reference")
    name, separator, digest = reference.partition("#sha256=")
    _require(separator and name == "contextual-finalization.json" and _hash(digest),
             "Invalid finalization reference")
    path = evidence.inside(directory, name)
    _require(_identity(path)["sha256"] == digest, "Finalization file digest mismatch")
    document = evidence.strict_json(path.read_bytes())
    _require(final == document and type(final) is dict, "Supplied finalization differs from its file")
    _require(final.get("schema") == "heroes-contextual-finalization-evidence-v1"
             and final.get("episode") == manifest["episode"]
             and final.get("identity") == manifest.get("identity")
             and type(final.get("identity")) is dict and bool(final["identity"])
             and type(final.get("returncode")) is int
             and final["returncode"] == manifest.get("returncode")
             and type(manifest.get("returncode")) is int
             and final.get("streams_complete") == [True, True]
             and all(type(v) is bool for v in final["streams_complete"])
             and final.get("engine_evidence_error") is None
             and final.get("terminal_barrier_error") is None
             and final.get("natural_terminal_after_cutoff") is None,
             "Finalization identity, stream completion or failure mismatch")
    _require(final.get("expected_stop_signals") == [15, 9]
             and all(type(v) is int for v in final["expected_stop_signals"]),
             "Unrecognized shutdown signals")
    continuation = final.get("continuation")
    exact(continuation, {"limit_per_player", "actions", "allowed_actions", "learner_excluded"},
          "continuation declaration")
    allowed = ["back", "close", "end_turn"] if profile != contract.WINDOWS_PROFILE else ["close", "end_turn"]
    _require(type(continuation["limit_per_player"]) is int
             and continuation["limit_per_player"] == CONTINUATION_LIMIT
             and type(continuation["actions"]) is list and len(continuation["actions"]) == 2
             and all(type(n) is int and 0 <= n <= CONTINUATION_LIMIT for n in continuation["actions"])
             and continuation["allowed_actions"] == allowed and continuation["learner_excluded"] is True,
             "Invalid bounded continuation declaration")
    exact(manifest.get("players"), set(COLORS), "player declarations")
    exact(final.get("raw_bridge_sha256"), set(COLORS), "bridge digest declarations")
    status = manifest.get("status")
    if status == "completed":
        outcomes = final.get("socket_outcomes")
        _require(outcomes in (["win", "loss"], ["loss", "win"])
                 and final.get("engine_verified_outcomes") == dict(zip(COLORS, outcomes))
                 and manifest.get("outcomes_by_player") == dict(zip(COLORS, outcomes))
                 and manifest.get("rewards_by_player") == {c: 1 if o == "win" else -1 for c, o in zip(COLORS, outcomes)}
                 and all(type(v) is int for v in manifest["rewards_by_player"].values())
                 and manifest.get("termination_reason") == "natural_terminal"
                 and final["returncode"] == 0 and final.get("reason") is None
                 and final.get("declared_stop") is False and final.get("shutdown_verified") is False
                 and continuation["actions"] == [0, 0], "Invalid natural finalization metadata")
    elif status == "truncated":
        _require(manifest.get("termination_reason") in HEADLESS_TRUNCATIONS
                 and final.get("reason") == manifest["termination_reason"]
                 and final.get("declared_stop") is True and final.get("shutdown_verified") is True
                 and final["returncode"] in (-15, -9) and final.get("socket_outcomes") == [None, None]
                 and final.get("engine_verified_outcomes") is None
                 and all(manifest.get(key) == dict.fromkeys(COLORS)
                         for key in ("outcomes_by_player", "rewards_by_player")),
                 "Invalid or scored cutoff metadata")
    else:
        raise ValueError("Failed/unfinalized episodes are excluded from history replay verification")
    return profile


def _reconstruct(directory, manifest, final, profile, player):
    color = COLORS[player]
    declaration = manifest["players"][color]
    _require(type(declaration) is dict and declaration.get("ready") is True
             and declaration.get("error") is None, f"{color}: failed/unready player")
    target = declaration.get(profile.history_key)
    exact(target, {"path", "sha256", "canonical_sha256", "prefix_sha256", "boundary", "complete"},
          f"{color} learner history declaration")
    expected_boundary = "natural" if manifest["status"] == "completed" else "cutoff"
    _require(target["complete"] is True and target["boundary"] == expected_boundary
             and all(_hash(target[key]) for key in ("sha256", "canonical_sha256", "prefix_sha256")),
             f"{color}: invalid learner boundary or hashes")
    path = evidence.inside(directory, f"agents/{color}/bridge.jsonl")
    bridge_identity = _identity(path)
    _require(bridge_identity["sha256"] == final["raw_bridge_sha256"][color], f"{color}: bridge digest mismatch")
    producer = ContextualProducer(manifest["episode"], player, interface=profile.environment)
    digest, counts, frame_count = hashlib.sha256(), Counter(), 0

    def appended(frames):
        nonlocal frame_count
        for frame in frames:
            digest.update(_wire(frame))
            counts[frame.kind] += 1
            if frame.kind == "acknowledged":
                counts["accepted" if frame.accepted else "rejected"] += 1
            frame_count += 1

    def locate_cutoff():
        # A zero-continuation cutoff can precede late native events. Locate the
        # frozen prefix from its declared digest, never from saved history rows.
        if (manifest["status"] != "truncated" or producer.cutoff is not None
                or producer.live.decision is None or producer.live.action is not None):
            return
        marker = contract.Frame(producer.episode, player, frame_count, producer.live.day, "cutoff",
                                decision_seq=producer.live.decision.seq, reason=final["reason"],
                                schema=profile.history_schema)
        candidate = digest.copy()
        candidate.update(_wire(marker))
        if candidate.hexdigest() == target["prefix_sha256"]:
            producer.freeze_cutoff(final["reason"])

    for row in _records(path):
        exact(row, {"t", "direction", "message"}, "bridge row")
        _require(type(row["t"]) in (float, int) and row["t"] >= 0
                 and (type(row["t"]) is int or math.isfinite(row["t"])),
                 f"{color}: invalid bridge timestamp")
        counts["rows"] += 1
        if row["direction"] == "engine":
            appended(producer.ingest(row["message"]))
            counts["engine_messages"] += 1
            locate_cutoff()
        elif row["direction"] == "policy":
            message = row["message"]
            exact(message, {"protocol_version", "type", "episode", "player", "decision_id", "revision",
                            "action", "delivered", "outside_learner_prefix"}, "policy delivery")
            _require(type(message["delivered"]) is bool and type(message["outside_learner_prefix"]) is bool,
                     f"{color}: invalid delivery flags")
            _require(all(type(message[key]) is int for key in ("protocol_version", "player", "decision_id", "revision")),
                     f"{color}: invalid response integer fields")
            # A failed send must fail the episode, even if a prior prefix looked complete.
            _require(message["delivered"], f"{color}: undelivered action in successful episode")
            _require(message["outside_learner_prefix"] == (producer.cutoff is not None),
                     f"{color}: continuation does not match the bound cutoff")
            decision = producer.live.decision
            _require(decision is not None, f"{color}: action without held decision")
            action = contract.decode_frame(dict(schema=profile.history_schema, episode=producer.episode,
                player=player, seq=frame_count, day=producer.live.day, kind="submitted",
                decision_seq=decision.seq, action=message["action"])).action
            expected = producer.response(action)
            actual = {key: value for key, value in message.items() if key not in ("delivered", "outside_learner_prefix")}
            _require(actual == expected, f"{color}: submitted response does not match its actual request")
            if producer.cutoff is not None:
                context = producer.live.context
                kind = ("end_turn" if context.kind == "map" else "back"
                        if context.kind == "stack" or context.panel in ("hall", "building", "recruitment", "tavern") else "close")
                neutral = contract.Action(kind, subject=context.subject if kind == "back" else None,
                                          schema=profile.action_schema)
                _require(action == neutral and producer.continuation_steps < CONTINUATION_LIMIT,
                         f"{color}: action is not the declared bounded technical continuation")
            appended((producer.submitted(action),))
            counts["delivered_actions"] += 1
        else:
            raise ValueError(f"{color}: unsupported bridge direction")
    _require(producer.ready and frame_count > 0, f"{color}: empty/incomplete player stream")
    if manifest["status"] == "truncated":
        _require(producer.cutoff is not None and producer.terminal is None,
                 f"{color}: no matching actual cutoff, or terminal after cutoff")
    _require(producer.history.digest() == target["prefix_sha256"], f"{color}: reconstructed prefix digest mismatch")
    _require(producer.terminal == final["socket_outcomes"][player]
             and declaration.get("terminal") == ({"outcome": producer.terminal} if producer.terminal else None),
             f"{color}: terminal declaration differs from transport")
    for key, expected in (("actions_sent", counts["delivered_actions"]), ("responses_undelivered", 0),
                          ("actions_accepted", counts["accepted"]), ("actions_rejected", counts["rejected"])):
        _require(type(declaration.get(key)) is int and declaration[key] == expected, f"{color}: incorrect {key}")
    _require(producer.continuation_steps == final["continuation"]["actions"][player],
             f"{color}: continuation count differs from transport")
    return producer, counts, bridge_identity


def _compare(directory, name, frames, digest):
    path = evidence.inside(directory, name)
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for frame in frames:
            line = stream.readline()
            _require(line == _wire(frame), f"History differs from raw bridge: {name}, sequence {frame.seq}")
            hasher.update(line)
            size += len(line)
        _require(stream.read(1) == b"", f"Extra history records: {name}")
    _require(hasher.hexdigest() == digest, f"History file digest mismatch: {name}")
    return {"sha256": hasher.hexdigest(), "bytes": size, "frames": len(frames)}


def verify_histories(directory, manifest, finalization_document):
    """Rebuild and compare a completed natural/cutoff pair, raising on any gap.

    The returned verified flag is scoped to history reconstruction. It is never
    a native replay or terminal-drain claim; callers must check those separately.
    """
    directory = Path(directory).resolve()
    final = finalization_document
    profile = _bound_metadata(directory, manifest, final)
    reconstructed = tuple(_reconstruct(directory, manifest, final, profile, player) for player in (0, 1))
    histories = tuple(item[0].history for item in reconstructed)
    if manifest["status"] == "completed":
        contract.finalize_natural(histories,
            engine_outcomes=tuple(final["engine_verified_outcomes"][color] for color in COLORS),
            socket_outcomes=tuple(final["socket_outcomes"]), process_exit_code=final["returncode"],
            evidence_ref=manifest["evidence_ref"], histories_complete=tuple(final["streams_complete"]))
    else:
        contract.finalize_cutoff(histories, evidence_ref=manifest["evidence_ref"],
            histories_complete=tuple(final["streams_complete"]), failure_free=True,
            shutdown_verified=final["shutdown_verified"], continuation_limit=final["continuation"]["limit_per_player"],
            continuation_steps=tuple(item[0].continuation_steps for item in reconstructed))
    from validation.probes.audit_contextual_histories import _dialog_cutoff_checks
    _dialog_cutoff_checks(manifest, histories)
    if profile.dialogs:
        _require("dialog_cutoff_drain" in final and final["dialog_cutoff_drain"] == manifest["dialog_cutoff_drain"],
                 "Finalization does not bind post-budget strategic choices")
    players = {}
    for color, (producer, counts, bridge) in zip(COLORS, reconstructed):
        declaration = manifest["players"][color]
        learner = declaration[profile.history_key]
        live = declaration.get("live_evidence")
        exact(live, {"path", "sha256", "continuation_steps", "learner_input"}, f"{color} live evidence")
        _require(learner["path"] == f"agents/{color}/{profile.history_filename}"
                 and live["path"] == f"agents/{color}/live-{profile.history_filename}"
                 and live["learner_input"] is False and type(live["continuation_steps"]) is int
                 and live["continuation_steps"] == producer.continuation_steps
                 and _hash(live["sha256"]), f"{color}: invalid live evidence declaration")
        _require(producer.history.digest() == learner["canonical_sha256"], f"{color}: canonical digest mismatch")
        players[color] = {"verified": True, "bridge": bridge, "counts": dict(counts),
            "continuation_steps": producer.continuation_steps,
            "cutoff_decision_seq": producer.cutoff.seq if producer.cutoff is not None else None,
            "learner": _compare(directory, learner["path"], producer.history.frames, learner["sha256"]),
            "live": _compare(directory, live["path"], producer.live.frames, live["sha256"])}
    return {"schema": "heroes-contextual-history-reconstruction-v1", "verified": True,
            "scope": "transport_to_history", "native_replay_verified": False,
            "episode": manifest["episode"], "boundary": histories[0].boundary, "players": players}
