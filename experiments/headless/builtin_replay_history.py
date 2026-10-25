"""Rebuild the sole learner history from mixed-game transport evidence.

No native opponent observations or paired finalization frames are synthesized.
Native command replay and terminal verification are separate admission checks.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

from experiments.headless import contextual_evidence as evidence
from experiments.headless.contextual_protocol import ContextualProducer, exact
from experiments.headless.contextual_replay_history import _compare, _records, _wire
from experiments.learner import contextual as contract

COLORS = ("red", "blue")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_history(directory, manifest, finalization_document):
    from experiments.headless import builtin_evidence as builtin
    directory = Path(directory).resolve()
    require(manifest == evidence.strict_json((directory / "manifest.json").read_bytes()), "Manifest differs from disk")
    final = builtin.read_finalization(directory, manifest)
    require(final == finalization_document, "Finalization differs from supplied document")
    player, profile = manifest["learner_player"], builtin.manifest_profile(manifest)
    require(type(player) is int and player in (0, 1), "Invalid learner seat")
    color = COLORS[player]
    exact(manifest["players"], {color}, "sole learner declaration")
    declaration = manifest["players"][color]
    require(declaration.get("ready") is True and declaration.get("error") is None, "Unready or failed learner stream")
    target = declaration[profile.history_key]
    exact(target, {"path", "sha256", "canonical_sha256", "prefix_sha256", "boundary", "complete"}, "learner history binding")
    require(target["complete"] is True and target["boundary"] == ("natural" if manifest["status"] == "completed" else "cutoff")
            and manifest["status"] in ("completed", "truncated"), "Unsupported history boundary")
    producer = ContextualProducer(manifest["episode"], player, interface=profile.environment)
    bridge_path = evidence.inside(directory, f"agents/{color}/bridge.jsonl")
    bridge = evidence.file_identity(bridge_path)
    require(final["raw_bridge_sha256"] == {color: bridge["sha256"]}, "Learner bridge digest differs")
    digest, counts, frames = hashlib.sha256(), Counter(), 0

    def appended(values):
        nonlocal frames
        for frame in values:
            digest.update(_wire(frame))
            frames += 1
            counts[frame.kind] += 1
            if frame.kind == "acknowledged":
                counts["accepted" if frame.accepted else "rejected"] += 1

    def locate_cutoff():
        if (manifest["status"] != "truncated" or producer.cutoff is not None
                or producer.live.decision is None or producer.live.action is not None
                or producer.live.context.kind == "dialog"):
            return
        marker = contract.Frame(producer.episode, player, frames, producer.live.day, "cutoff",
            decision_seq=producer.live.decision.seq, reason=manifest["termination_reason"], schema=profile.history_schema)
        candidate = digest.copy()
        candidate.update(_wire(marker))
        if candidate.hexdigest() == target["prefix_sha256"]:
            producer.freeze_cutoff(manifest["termination_reason"])

    for row in _records(bridge_path):
        exact(row, {"t", "direction", "message"}, "bridge row")
        require(type(row["t"]) in (int, float) and row["t"] >= 0 and math.isfinite(row["t"]), "Invalid bridge timestamp")
        counts["rows"] += 1
        if row["direction"] == "engine":
            appended(producer.ingest(row["message"]))
            counts["engine_messages"] += 1
            locate_cutoff()
        elif row["direction"] == "policy":
            message = row["message"]
            exact(message, {"protocol_version", "type", "episode", "player", "decision_id", "revision",
                "action", "delivered", "outside_learner_prefix"}, "learner delivery")
            require(message["delivered"] is True and message["outside_learner_prefix"] is False
                    and producer.cutoff is None, "Unverified delivery or action after frozen learner prefix")
            require(all(type(message[key]) is int for key in ("protocol_version", "player", "decision_id", "revision")), "Invalid response identity types")
            decision = producer.live.decision
            require(decision is not None, "Response without actual held decision")
            action = contract.decode_frame(dict(schema=profile.history_schema, episode=producer.episode,
                player=player, seq=frames, day=producer.live.day, kind="submitted", decision_seq=decision.seq,
                action=message["action"])).action
            expected = producer.response(action)
            actual = {key: value for key, value in message.items() if key not in ("delivered", "outside_learner_prefix")}
            require(actual == expected, "Submitted learner response does not match its request")
            appended((producer.submitted(action),))
            counts["delivered_actions"] += 1
        else:
            raise ValueError("Unknown bridge direction")
    require(producer.ready and frames > 0, "Incomplete learner stream")
    if manifest["status"] == "truncated":
        require(producer.cutoff is not None and producer.terminal is None, "Missing cutoff or terminal after cutoff")
    else:
        require(producer.cutoff is None and producer.terminal in ("win", "loss"), "Missing actual learner terminal")
    require(producer.terminal == final["socket_outcome"]
            and declaration.get("terminal") == ({"outcome": producer.terminal} if producer.terminal else None), "Terminal differs from learner transport")
    for key, expected in (("actions_sent", counts["delivered_actions"]), ("responses_undelivered", 0),
        ("actions_accepted", counts["accepted"]), ("actions_rejected", counts["rejected"])):
        require(type(declaration.get(key)) is int and declaration[key] == expected, "Learner counter differs: " + key)
    require(producer.continuation_steps == 0, "Unexpected technical continuation")
    require(producer.history.digest() == target["prefix_sha256"] == target["canonical_sha256"], "Reconstructed learner history hash differs")
    builtin.validate_manifest_history(producer.history, manifest)
    live = declaration.get("live_evidence")
    exact(live, {"path", "sha256", "continuation_steps", "learner_input"}, "live evidence")
    require(target["path"] == f"agents/{color}/{profile.history_filename}"
            and live["path"] == f"agents/{color}/live-{profile.history_filename}"
            and live["learner_input"] is False and type(live["continuation_steps"]) is int
            and live["continuation_steps"] == 0, "Unexpected learner/live evidence declaration")
    return {"schema": "heroes-builtin-history-reconstruction-v1", "verified": True,
        "scope": "sole_learner_transport_to_history", "native_replay_verified": False,
        "episode": manifest["episode"], "learner_player": player, "boundary": target["boundary"],
        "bridge": bridge, "counts": dict(counts), "opponent_history": None,
        "cutoff_decision_seq": producer.cutoff.seq if producer.cutoff else None,
        "learner": _compare(directory, target["path"], producer.history.frames, target["sha256"]),
        "live": _compare(directory, live["path"], producer.live.frames, live["sha256"])}
