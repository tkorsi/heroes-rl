#!/usr/bin/env python3
"""Read-only audit of two persisted contextual histories and their manifest.

Reconstructs the strict contract independently of the live adapter. A successful
audit establishes internal record consistency, not native fog correctness or
that every engine event was captured. Without a manifest, eligibility is false.
No games, neural inference, training, or source changes are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.learner import contextual as contract

COLORS = ("red", "blue")
FINALIZED_KINDS = ("finalized", "cutoff_finalized")
SCOPE = ("Record shape, chronology, immutable prefixes and manifest binding only. "
         "Observed coverage does not prove source visibility, event completeness, "
         "engine outcomes, process provenance or replay fidelity.")


def _canonical(frame):
    return json.dumps(contract.encode_frame(frame), sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _digest(frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(_canonical(frame))
    return digest.hexdigest()


def _closing_proof(history):
    frame = history.frames[-1]
    if frame.kind == "finalized":
        return frame.verification
    if frame.kind == "cutoff_finalized":
        return frame.cutoff_verification
    return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))


def _sightings(frames):
    decisions = [frame for frame in frames if frame.kind == "decision"]
    transient, intermediate, intervals = [], [], set()

    def enemies(frame):
        if frame.observation is None:
            return {}
        return {obj.handle: obj.position for obj in frame.observation.objects
                if obj.owner == "opponent" and obj.kind == 34}

    for before, after in zip(decisions, decisions[1:]):
        left, right, seen = enemies(before), enemies(after), set()
        for frame in frames[before.seq + 1:after.seq]:
            if frame.kind != "view":
                continue
            for handle, position in enemies(frame).items():
                if (handle, position) in seen:
                    continue
                seen.add((handle, position))
                sample = {"decision_before_seq": before.seq, "decision_after_seq": after.seq,
                          "sighting_seq": frame.seq, "day": frame.day,
                          "handle": handle, "position": list(position)}
                if handle not in left and handle not in right:
                    transient.append(sample)
                    intervals.add((before.seq, after.seq))
                if left.get(handle) != position and right.get(handle) != position:
                    intermediate.append(sample)
    return {"fully_transient_position_samples": len(transient),
            "fully_transient_decision_intervals": len(intervals),
            "intermediate_position_samples": len(intermediate),
            "fully_transient_examples": transient[:8], "intermediate_examples": intermediate[:8]}


def _coverage(frames):
    observations = [frame.observation for frame in frames if frame.observation is not None]
    tiles = [tile for observation in observations for tile in observation.tiles]
    return {
        "frame_kinds": dict(sorted(Counter(frame.kind for frame in frames).items())),
        "submitted_action_kinds": dict(sorted(Counter(frame.action.kind for frame in frames if frame.action).items())),
        "result_statuses": dict(sorted(Counter(frame.result.status for frame in frames if frame.result).items())),
        "consumption_layers": dict(sorted(Counter(frame.consumer for frame in frames if frame.consumer).items())),
        "native_acknowledgements": {"accepted": sum(frame.kind == "acknowledged" and frame.accepted is True for frame in frames),
                                     "rejected": sum(frame.kind == "acknowledged" and frame.accepted is False for frame in frames)},
        "observation_contexts": dict(sorted(Counter(obs.context.kind for obs in observations).items())),
        "window_subjects": sorted({obs.window.context.subject for obs in observations if obs.window}),
        "tile_fact_samples": len(tiles), "road_field_samples": len(tiles),
        "road_id_values": sorted({tile.road for tile in tiles}),
        "distinct_observed_tile_positions": len({tile.position for tile in tiles}),
        "opponent_sightings": _sightings(frames),
        "source_visibility_proven": False,
        "closed_window_nonrefresh_proven": False,
        "coverage_note": "Counts describe decoded records only; native paired-state checks are separate evidence.",
    }


def audit_player(path, player):
    path = Path(path).resolve()
    report = {"path": str(path), "player": player, "color": COLORS[player],
              "failures": [], "record_contract_eligible": False,
              "has_eligible_samples": False, "learner_eligible": False}
    history = None
    frozen_encodings = []
    try:
        data = path.read_bytes()
        report["file_sha256"] = hashlib.sha256(data).hexdigest()
        lines = data.decode("utf-8").splitlines()
        report["file_lines"] = len(lines)
        if not lines:
            raise ValueError("Empty history")
        for number, line in enumerate(lines, 1):
            try:
                if not line.strip():
                    raise ValueError("Blank history record")
                raw = _json(line)
                # The serialized envelope must carry its version, even if a
                # dataclass constructor has a convenient in-process default.
                if type(raw) is not dict or raw.get("schema") not in (contract.SCHEMA, contract.TOWN_SCHEMA):
                    raise ValueError("Explicit supported contextual history schema is required")
                frame = contract.decode_frame(raw)
                if frame.player != player:
                    raise ValueError("History belongs to the wrong player")
                if history is None:
                    history = contract.PlayerHistory(frame.episode, player, schema=frame.schema)
                history.append(frame)
                frozen_encodings.append(_canonical(frame))
            except Exception as error:
                report["failures"].append({"phase": "decode_append", "line": number, "error": str(error)})
                break
        if history is None:
            return report, None
        frames = history.frames
        if any(_canonical(frame) != previous for frame, previous in zip(frames, frozen_encodings)):
            raise ValueError("A previously appended immutable prefix changed")
        report.update(episode=history.episode, history_schema=history.schema, environment=history.profile.environment,
                      decoded_frames=len(frames), boundary=history.boundary,
                      terminal=history.terminal, canonical_sha256=_digest(frames),
                      prefix_sha256=_digest(frames[:-1] if frames and frames[-1].kind in FINALIZED_KINDS else frames),
                      coverage=_coverage(frames))
        if history.digest() != report["canonical_sha256"]:
            raise ValueError("Independent canonical history digest disagrees with contract")
        transitions = history.transitions()
        report["transitions"] = {
            "completed": len(transitions),
            "contract_eligible": sum(t.eligible for t in transitions),
            "contract_bootstrap_allowed": sum(t.bootstrap_allowed for t in transitions),
            "contract_terminal_rewards": [t.reward for t in transitions if t.reward in (-1, 1)],
            "effective_eligible": 0, "effective_bootstrap_allowed": 0,
        }
        if history.boundary not in ("natural", "cutoff", "failure"):
            report["failures"].append({"phase": "boundary", "error": "History has no explicit final boundary"})
        if history.boundary == "failure" and any(t.eligible or t.bootstrap_allowed or t.reward is not None for t in transitions):
            report["failures"].append({"phase": "targets", "error": "Failed history supplies targets or bootstrap"})
    except Exception as error:
        report["failures"].append({"phase": "history", "error": str(error)})
    return report, history


def _is_digest(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _manifest_checks(manifest, reports, histories):
    failures = []

    def require(condition, message):
        if not condition:
            failures.append({"phase": "manifest", "error": message})

    if type(manifest) is not dict:
        require(False, "Manifest must be an object")
        return failures
    profile = histories[0].profile
    expected = {"schema_version": 1, "kind": "contextual-headless-self-play", "environment": profile.environment,
                "history_schema": profile.history_schema, "action_schema": profile.action_schema}
    for key, value in expected.items():
        require(type(manifest.get(key)) is type(value) and manifest.get(key) == value, f"Invalid manifest {key}")
    require(all(manifest.get("episode") == history.episode for history in histories), "Manifest episode mismatch")
    for name in ("termination_reason", "evidence_ref"):
        require(type(manifest.get(name)) is str and bool(manifest.get(name, "").strip()), f"Missing manifest {name}")
    require(type(manifest.get("returncode")) is int, "Manifest returncode must be an integer")
    identity = manifest.get("identity", {})
    if type(identity) is not dict:
        require(False, "Manifest identity must be an object")
        identity = {}
    require(type(identity.get("engine_revision")) is str and bool(identity.get("engine_revision", "").strip()),
            "Missing engine revision provenance")
    for name in ("bridge_source_sha256", "map_sha256"):
        require(_is_digest(identity.get(name)), f"Missing or invalid identity.{name}")
    status = manifest.get("status")
    require(type(status) is str and status in ("completed", "truncated", "failed"), "Unsupported manifest status")
    expected_boundary = {"completed": "natural", "truncated": "cutoff", "failed": "failure"}.get(status) if type(status) is str else None
    require(all(history.boundary == expected_boundary for history in histories), "Manifest status/history boundary mismatch")
    outcomes, rewards = manifest.get("outcomes_by_player"), manifest.get("rewards_by_player")
    if status == "completed":
        actual = {color: history.terminal for color, history in zip(COLORS, histories)}
        require(tuple(actual.values()) in (("win", "loss"), ("loss", "win")), "Natural outcomes are not complementary")
        require(outcomes == actual, "Manifest natural outcomes disagree with histories")
        expected_rewards = {color: 1 if outcome == "win" else -1 for color, outcome in actual.items()}
        require(rewards == expected_rewards and type(rewards) is dict and all(type(v) is int for v in rewards.values()),
                "Manifest natural rewards disagree with outcomes")
        require(type(manifest.get("returncode")) is int and manifest.get("returncode") == 0, "Natural manifest requires clean exit")
    else:
        require(outcomes == {color: None for color in COLORS}, "Unscored manifest must have null outcomes")
        require(rewards == {color: None for color in COLORS}, "Unscored manifest must have null rewards")
    if status in ("completed", "truncated"):
        proof = _closing_proof(histories[0])
        require(proof is not None and proof.evidence_ref == manifest.get("evidence_ref"), "Manifest finalization evidence reference mismatch")
    players = manifest.get("players", {})
    if type(players) is not dict:
        require(False, "Manifest players must be an object")
        players = {}
    for color, report, history in zip(COLORS, reports, histories):
        player = players.get(color, {})
        declaration = player.get(profile.history_key, {}) if type(player) is dict else {}
        other_key = contract.TOWN_PROFILE.history_key if profile == contract.WINDOWS_PROFILE else contract.WINDOWS_PROFILE.history_key
        require(type(player) is dict and other_key not in player, f"{color} contains a different history version declaration")
        if type(declaration) is not dict:
            declaration = {}
        for name, derived in (("sha256", "file_sha256"), ("prefix_sha256", "prefix_sha256")):
            require(declaration.get(name) == report.get(derived), f"{color} {profile.history_key}.{name} mismatch")
        if "canonical_sha256" in declaration:
            require(declaration["canonical_sha256"] == report.get("canonical_sha256"), f"{color} canonical history hash mismatch")
        require(declaration.get("boundary") == history.boundary, f"{color} declared history boundary mismatch")
        require(type(declaration.get("complete")) is bool, f"{color} history completeness must be explicit boolean")
        if status in ("completed", "truncated"):
            require(declaration.get("complete") is True, f"{color} scored/cutoff history must be complete")
    return failures


def audit(red, blue, manifest=None):
    """Return JSON-safe diagnostics; manifest is a filename or parsed object."""
    pairs = [audit_player(path, player) for player, path in enumerate((red, blue))]
    reports, histories = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    failures = [{"player": COLORS[index], **failure} for index, report in enumerate(reports) for failure in report["failures"]]
    manifest_data = None
    manifest_description = None
    if manifest is not None:
        try:
            if isinstance(manifest, (str, Path)):
                path = Path(manifest).resolve()
                data = path.read_bytes()
                manifest_data = _json(data)
                manifest_description = {"path": str(path), "file_sha256": hashlib.sha256(data).hexdigest()}
            else:
                manifest_data = manifest
                manifest_description = {"source": "provided_object"}
            if type(manifest_data) is not dict:
                raise ValueError("Provided manifest must be a JSON object")
        except Exception as error:
            failures.append({"phase": "manifest_read", "error": str(error)})
    if all(history is not None and history.frames for history in histories):
        if histories[0].episode != histories[1].episode:
            failures.append({"phase": "pair", "error": "Different player episode identities"})
        if histories[0].schema != histories[1].schema:
            failures.append({"phase": "pair", "error": "Different player history schemas"})
        if histories[0].boundary != histories[1].boundary:
            failures.append({"phase": "pair", "error": "Both player histories require the same final boundary"})
        if any(history.boundary in ("natural", "cutoff") for history in histories):
            proofs = [_closing_proof(history) for history in histories]
            prefixes = tuple(report.get("prefix_sha256") for report in reports)
            if any(proof is None for proof in proofs) or proofs[0] != proofs[1] or any(
                    proof is not None and proof.history_sha256 != prefixes for proof in proofs):
                failures.append({"phase": "pair", "error": "Both final proofs must bind the same two complete prefixes"})
        if manifest_data is not None:
            try:
                failures.extend(_manifest_checks(manifest_data, reports, histories))
            except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
                failures.append({"phase": "manifest", "error": "Malformed manifest: " + str(error)})
    consistent = not failures
    manifest_bound = consistent and manifest_data is not None
    for report, history in zip(reports, histories):
        record_eligible = manifest_bound and history.boundary in ("natural", "cutoff")
        report["record_contract_eligible"] = record_eligible
        if record_eligible:
            report["transitions"]["effective_eligible"] = report["transitions"]["contract_eligible"]
            report["transitions"]["effective_bootstrap_allowed"] = report["transitions"]["contract_bootstrap_allowed"]
        report["has_eligible_samples"] = report.get("transitions", {}).get("effective_eligible", 0) > 0
        report["learner_eligible"] = record_eligible and report["has_eligible_samples"]
    record_eligible = all(report["record_contract_eligible"] for report in reports)
    has_samples = any(report["has_eligible_samples"] for report in reports)
    return {"audit_schema": "heroes-contextual-history-audit-v1", "consistent": consistent,
            "manifest_bound": manifest_bound, "scoring_binding_verified": manifest_bound and manifest_data.get("status") == "completed",
            "record_contract_eligible": record_eligible, "has_eligible_samples": has_samples,
            "learner_eligible": record_eligible and has_samples,
            "eligibility_semantics": "Record eligibility requires both valid bound histories. Learner eligibility additionally requires at least one effective eligible sample; inspect each player's counts separately.",
            "scope": SCOPE, "source_visibility_proven": False, "training_performed": False,
            "manifest": manifest_description, "players": dict(zip(COLORS, reports)), "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red", type=Path, required=True, help="Red player history-v2 JSONL")
    parser.add_argument("--blue", type=Path, required=True, help="Blue player history-v2 JSONL")
    parser.add_argument("--manifest", type=Path, help="Required for scoring/eligibility binding; otherwise structural audit only")
    parser.add_argument("--output", type=Path, help="Write a new JSON report; existing files are refused")
    args = parser.parse_args(argv)
    report = audit(args.red, args.blue, args.manifest)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(encoded)
    else:
        sys.stdout.write(encoded)
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
