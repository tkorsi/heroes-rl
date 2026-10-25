"""One controlled player's evaluation boundary; never a fabricated AI trajectory.

The unchanged v7 player stream has no paired finalization frame. This separate
envelope binds its complete prefix to native completion or an explicit cutoff.
It is diagnostic evidence and is never passed to the policy or value model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.headless import contextual_evidence as native
from experiments.learner import contextual as c

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

MANIFEST_KIND = "contextual-builtin-evaluation"
FINAL_SCHEMA = "heroes-contextual-builtin-finalization-v1"
REPLAY_SCHEMA = "heroes-contextual-builtin-replay-v1"
CONTROLLERS_SCHEMA = "heroes-contextual-controllers-v1"
COLORS = ("red", "blue")
HORIZONS = {"collection_window", "game_day_limit", "agent_action_limit"}
ADAPTER_SOURCES = native.ADAPTER_SOURCES | {
    "experiments/headless/builtin_env.py", "experiments/headless/builtin_evidence.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def controllers(learner_player):
    require(type(learner_player) is int and learner_player in (0, 1), "Expected one Red or Blue learner")
    return {"schema": CONTROLLERS_SCHEMA, "explorer_players": [learner_player],
            "native_opponents": [{"player": 1 - learner_player, "module": "Nullkiller"}]}


def opponent_identity(learner_player, binary_sha256):
    controllers(learner_player)
    require(type(binary_sha256) is str and len(binary_sha256) == 64
            and all(x in "0123456789abcdef" for x in binary_sha256), "Expected actual Nullkiller library hash")
    return {"kind": "builtin", "module": "Nullkiller", "player": 1 - learner_player,
            "difficulty": 0, "binary_sha256": binary_sha256}


def capture_files(directory):
    result = native.capture_files(directory)
    return {**result, "schema": REPLAY_SCHEMA}


def terminal_barrier(directory, *, episode, interface, outcomes, learner_player):
    expected_controllers = controllers(learner_player)
    require(type(episode) is str and bool(episode) and interface == c.DIALOGS_ENVIRONMENT
            and outcomes in (["win", "loss"], ["loss", "win"]), "Invalid mixed terminal identity")
    records, files = [], {}
    for side, name in zip(("server", "client"), native.FILES):
        path = native.inside(directory, name)
        data = path.read_bytes()
        require(data.endswith(b"\n") and len(data.splitlines()) == 1, "Expected one complete terminal receipt")
        record = native.strict_json(data)
        fields = native.COMMON | {"controllers"}
        if side == "client":
            fields |= {"callbacks_complete", "observed_count", "native_terminal_callbacks", "explorer_terminal_callbacks"}
        require(type(record) is dict and set(record) == fields, "Unexpected mixed terminal receipt fields")
        require(record["schema"] == native.SCHEMA and record["kind"] == side
                and record["episode"] == episode and record["interface"] == interface
                and type(record["barrier_id"]) is int and record["barrier_id"] == 1
                and record["origin"] == "request" and record["outcomes"] == outcomes
                and same(record["controllers"], expected_controllers), "Mixed terminal receipt identity mismatch")
        request = record["request"]
        require(type(request) is dict and set(request) == {"player", "id", "type"}
                and type(request["player"]) is int
                and (request["player"] in (0, 1) or request["player"] == -1 and request["type"] == 198)
                and type(request["id"]) is int and 0 <= request["id"] <= 0xffffffff
                and type(request["type"]) is int and 1 <= request["type"] <= 0xffff,
                "Invalid mixed terminal request origin")
        require(type(record["client_packets"]) is int and 0 < record["client_packets"] <= 0x7fffffffffffffff,
                "Invalid terminal packet position")
        if side == "client":
            require(record["callbacks_complete"] is True and type(record["observed_count"]) is int
                    and record["observed_count"] == record["client_packets"]
                    and record["native_terminal_callbacks"] == [0, 1]
                    and record["explorer_terminal_callbacks"] == [learner_player]
                    and all(type(v) is int for v in record["native_terminal_callbacks"] + record["explorer_terminal_callbacks"]),
                    "Mixed terminal callbacks are incomplete")
        records.append(record)
        files[name] = native.file_identity(path)
    require(all(records[0][key] == records[1][key] for key in (native.COMMON | {"controllers"}) - {"kind"}),
            "Server/client mixed terminal receipts differ")
    return {"schema": native.SCHEMA, "verified": True, "files": files, "server": records[0], "client": records[1]}


def _drain(history, manifest):
    require("dialog_cutoff_drain" in manifest, "Missing explicit dialog cutoff declaration")
    drain = manifest["dialog_cutoff_drain"]
    if drain is None:
        return
    require(type(drain) is dict and set(drain) == {"reason", "trigger", "limit", "choices", "learner_included"}
            and drain["reason"] in HORIZONS and type(drain["limit"]) is int and drain["limit"] == 8
            and drain["learner_included"] is True, "Invalid single-player dialog drain")
    trigger, frames = drain["trigger"], history.frames
    require(type(trigger) is dict and set(trigger) == {"player", "decision_seq", "actions_sent"}
            and type(trigger["player"]) is int and trigger["player"] == history.player
            and type(trigger["decision_seq"]) is int and 0 <= trigger["decision_seq"] < len(frames)
            and type(trigger["actions_sent"]) is int and trigger["actions_sent"] >= 0,
            "Invalid single-player drain trigger")
    seq = trigger["decision_seq"]
    require(frames[seq].kind == "decision" and frames[seq].observation.context.kind == "dialog"
            and sum(f.kind == "submitted" for f in frames[:seq]) == trigger["actions_sent"],
            "Drain trigger does not bind its actual own dialog")
    expected = [{"player": history.player, "decision_seq": f.decision_seq}
                for f in frames[seq:] if f.kind == "submitted"]
    require(type(drain["choices"]) is list and same(drain["choices"], expected) and len(expected) <= 8
            and all(f.action.kind == "choose_dialog" for f in frames[seq:] if f.kind == "submitted"),
            "Deferred cutoff contains missing, excess or fabricated policy answers")


def validate_manifest_history(history, manifest):
    """Validate one genuine history and its external boundary, without an engine claim."""
    require(type(history) is c.PlayerHistory and history.schema == c.DIALOGS_SCHEMA and history.frames,
            "Expected a nonempty v7 own history")
    player, color = history.player, COLORS[history.player]
    expected = {"schema_version": 1, "kind": MANIFEST_KIND, "episode": history.episode,
                "environment": c.DIALOGS_ENVIRONMENT, "history_schema": c.DIALOGS_SCHEMA,
                "action_schema": c.DIALOGS_ACTION_SCHEMA, "learner_player": player}
    require(type(manifest) is dict and all(type(manifest.get(k)) is type(v) and manifest[k] == v
            for k, v in expected.items()), "Single-player manifest identity mismatch")
    require(same(manifest.get("controllers"), controllers(player)), "Mixed controller declaration changed")
    opponent = manifest.get("opponent_identity")
    require(type(opponent) is dict and same(opponent, opponent_identity(player, opponent.get("binary_sha256"))),
            "Missing actual native opponent identity")
    require(type(manifest.get("players")) is dict and set(manifest["players"]) == {color},
            "A built-in opponent cannot acquire an external-agent history")
    require(not any(f.kind in ("finalized", "cutoff_finalized") for f in history.frames),
            "Single-player evaluation cannot manufacture paired finalization")
    status = manifest.get("status")
    boundary = {"completed": "natural", "truncated": "cutoff", "failed": "failure"}.get(status)
    require(boundary is not None, "Unsupported single-player status")
    declaration = manifest["players"][color].get("history_v7", {})
    sha = history.digest()
    require(declaration.get("canonical_sha256") == sha and declaration.get("prefix_sha256") == sha
            and declaration.get("boundary") == boundary, "Single-player history hash or boundary changed")
    _drain(history, manifest)
    outcomes, rewards = manifest.get("outcomes_by_player"), manifest.get("rewards_by_player")
    require(type(outcomes) is dict and set(outcomes) == set(COLORS)
            and type(rewards) is dict and set(rewards) == set(COLORS), "Missing explicit game outcomes/rewards")
    if status == "completed":
        require(history.boundary is None and history.terminal in ("win", "loss")
                and outcomes[color] == history.terminal and list(outcomes[c] for c in COLORS) in
                (["win", "loss"], ["loss", "win"])
                and rewards == {c: 1 if o == "win" else -1 for c, o in outcomes.items()}
                and all(type(v) is int for v in rewards.values())
                and manifest.get("returncode") == 0 and type(manifest.get("returncode")) is int
                and manifest.get("termination_reason") == "natural_terminal",
                "Natural evaluation needs its actual own terminal and complementary native results")
    else:
        require(outcomes == dict.fromkeys(COLORS) and rewards == dict.fromkeys(COLORS), "Unscored evaluation received a result")
        if status == "truncated":
            require(history.boundary == "cutoff_pending" and history.frames[-1].kind == "cutoff"
                    and history.context.kind != "dialog" and history.action is None
                    and manifest.get("termination_reason") in HORIZONS,
                    "Cutoff requires the actual unacted safe own decision")
        else:
            require(history.boundary == "failure", "Failed evaluation lacks its failure boundary")
    require(type(declaration.get("complete")) is bool and (status == "failed" or declaration["complete"]),
            "Unfinished controlled stream")
    return {"consistent": True, "player": player, "frames": len(history.frames), "history_sha256": sha,
            "boundary": boundary, "training_samples": 0, "optimization_eligible": False,
            "native_execution_verified": False}


def _in_process(manifest):
    identity = manifest.get("identity")
    backend = identity.get("backend") if type(identity) is dict else None
    return type(backend) is dict and backend.get("kind") == "embedded-in-process-v1"


def _host_stop_proven(manifest, final):
    """The game process's signal status, or the in-process engine's close report."""
    if _in_process(manifest):
        from experiments.headless import embedded_evidence
        return embedded_evidence.truncation_closed(final)
    return final.get("returncode") in (-15, -9)


def _natural_close_proven(manifest, final):
    if _in_process(manifest):
        from experiments.headless import embedded_evidence
        return embedded_evidence.natural_closed(final)
    return True


def read_finalization(directory, manifest):
    """Bind the external proof; replay additionally verifies native semantics/source."""
    name, marker, sha = manifest.get("evidence_ref", "").partition("#sha256=")
    require(marker and name == "builtin-finalization.json", "Missing one-seat finalization reference")
    path = native.inside(directory, name)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha, "One-seat finalization bytes changed")
    final = native.strict_json(data)
    from experiments.headless import embedded_evidence
    embedded_evidence.require_consistent_backend(manifest, final)
    player, color = manifest["learner_player"], COLORS[manifest["learner_player"]]
    require(final.get("schema") == FINAL_SCHEMA and type(final.get("learner_player")) is int
            and final["learner_player"] == player
            and same(final.get("controllers"), controllers(player)) and final.get("episode") == manifest["episode"]
            and final.get("returncode") == manifest["returncode"] and final.get("identity") == manifest["identity"]
            and final.get("native_capture") == manifest["replay"]
            and final.get("dialog_cutoff_drain") == manifest["dialog_cutoff_drain"]
            and final.get("history_sha256") == manifest["players"][color]["history_v7"]["canonical_sha256"],
            "Finalization does not bind the one-seat manifest")
    if manifest["status"] in ("completed", "truncated"):
        require(final.get("stream_complete") is True and final.get("engine_evidence_error") is None,
                "Finalization lacks complete own stream/effective engine evidence")
    if manifest["status"] == "completed":
        require(final.get("engine_verified_outcomes") == manifest["outcomes_by_player"]
                and final.get("socket_outcome") == manifest["outcomes_by_player"][color]
                and final.get("terminal_barrier_error") is None and final.get("declared_stop") is False
                and _natural_close_proven(manifest, final),
                "Natural finalization has incomplete or interrupted terminal evidence")
    elif manifest["status"] == "truncated":
        require(final.get("declared_stop") is True and final.get("shutdown_verified") is True
                and _host_stop_proven(manifest, final) and final.get("reason") == manifest["termination_reason"]
                and final.get("engine_verified_outcomes") is None, "Unverified single-player cutoff shutdown")
    return final
