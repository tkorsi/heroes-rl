"""Independent native contextual replay and source-bound transport verification.

The native verifier regenerates server packets and runs the actual client and
adapter over them. This layer binds that check to the two delivered histories,
their action streams, completion evidence, map, code and executable identities.
It never changes a run manifest or promotes historical unanchored recordings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.headless import contextual_evidence as evidence
from experiments.headless import replay as legacy
from experiments.headless.contextual_protocol import declared_army_information
from experiments.headless.army_replay_mutations import army_management_mutations
from experiments.headless.dialog_replay_mutations import adventure_dialogs_mutations
from experiments.learner import contextual as contract

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-contextual-replay-v1"
REPORT_SCHEMA = "heroes-contextual-replay-verification-v1"
COLORS = ("red", "blue")
ReplayError = legacy.ReplayError


def records(path, *, cutoff=False):
    data = Path(path).read_bytes()
    end = data.rfind(b"\n") + 1
    if end < len(data) and not cutoff:
        raise ReplayError(f"Incomplete final record: {path}")
    rows = [evidence.strict_json(line) for line in data[:end].splitlines()]
    if any(type(row) is not dict for row in rows):
        raise ReplayError(f"Non-object record: {path}")
    return rows, len(data) - end


def bound_evidence(episode, manifest, binary):
    recorded = manifest.get("replay", {})
    if (recorded.get("schema") != SCHEMA or recorded.get("recorded") is not True
            or recorded.get("missing")):
        raise ReplayError("Contextual replay is unsupported for this unanchored recording")
    present = evidence.capture_files(episode)
    if present["files"] != recorded.get("files") or not present["recorded"]:
        raise ReplayError("Native contextual evidence changed after finalization")
    name, separator, digest = manifest.get("evidence_ref", "").partition("#sha256=")
    if not separator or name != "contextual-finalization.json":
        raise ReplayError("Missing source-bound contextual finalization")
    path = evidence.inside(episode, name)
    if evidence.file_identity(path)["sha256"] != digest:
        raise ReplayError("Contextual finalization digest mismatch")
    final = evidence.strict_json(path.read_bytes())
    identity = manifest.get("identity", {})
    if (final.get("identity") != identity or final.get("native_capture") != recorded
            or final.get("episode") != manifest.get("episode")
            or final.get("returncode") != manifest.get("returncode")
            or final.get("streams_complete") != [True, True]
            or final.get("engine_evidence_error") is not None):
        raise ReplayError("Finalization does not bind the recorded episode")
    if contract.profile_for(manifest["environment"]).dialogs:
        if "dialog_cutoff_drain" not in manifest or final.get("dialog_cutoff_drain") != manifest["dialog_cutoff_drain"]:
            raise ReplayError("Finalization does not bind strategic dialog cutoff choices")
    if identity.get("captured_before_launch") is not True:
        raise ReplayError("Native identities were not captured before launch")
    runtime = identity.get("native_runtime_sha256", {})
    if runtime != evidence.runtime_identity(binary.parent) or runtime.get("vcmireplay") != evidence.file_identity(binary)["sha256"]:
        raise ReplayError("Native runtime inventory or digests differ from the captured build")
    if identity.get("binary_sha256") != evidence.file_identity(binary.parent / "vcmiclient")["sha256"]:
        raise ReplayError("Native client differs from the captured build")
    explorer = next((value for name, value in runtime.items() if name in ("AI/libExplorerRL.dylib", "AI/libExplorerRL.so")), None)
    if not explorer or identity.get("explorer_rl_sha256") != explorer:
        raise ReplayError("Adapter binary identity mismatch")
    if identity.get("engine_recipe") != legacy.scoring.engine_recipe_digests(ROOT):
        raise ReplayError("Canonical native sources differ from the captured recipe")
    if set(identity.get("contextual_python_sha256", {})) != evidence.ADAPTER_SOURCES:
        raise ReplayError("Incomplete contextual adapter source inventory")
    for name, digest in identity["contextual_python_sha256"].items():
        if evidence.file_identity(evidence.inside(ROOT, name))["sha256"] != digest:
            raise ReplayError(f"Contextual adapter source changed: {name}")
    lock = evidence.strict_json((ROOT / "dependencies/vcmi.json").read_bytes())
    if identity.get("engine_revision") != lock["commit"] or identity.get("engine", {}).get("commit") != lock["commit"]:
        raise ReplayError("Pinned upstream engine identity mismatch")
    log = evidence.inside(episode, "logs/process.log")
    if evidence.file_identity(log)["sha256"] != final.get("process_log_sha256"):
        raise ReplayError("Process log changed after finalization")
    for color in COLORS:
        path = evidence.inside(episode, f"agents/{color}/bridge.jsonl")
        if evidence.file_identity(path)["sha256"] != final.get("raw_bridge_sha256", {}).get(color):
            raise ReplayError(f"{color} bridge changed after finalization")
    if manifest.get("status") == "completed":
        outcomes = [manifest.get("outcomes_by_player", {}).get(color) for color in COLORS]
        barrier = evidence.terminal_barrier(episode, episode=manifest["episode"],
                                           interface=manifest["environment"], outcomes=outcomes)
        evidence.terminal_capture_close(episode, barrier)
        if barrier != final.get("terminal_barrier") or final.get("terminal_barrier_error") is not None:
            raise ReplayError("Finalization terminal barrier mismatch")
        if (manifest.get("returncode") != 0 or manifest.get("termination_reason") != "natural_terminal"
                or final.get("engine_verified_outcomes") != manifest["outcomes_by_player"]
                or final.get("socket_outcomes") != outcomes
                or manifest.get("rewards_by_player") != {c: 1 if o == "win" else -1 for c, o in zip(COLORS, outcomes)}):
            raise ReplayError("Natural outcome labels disagree with barrier evidence")
    elif manifest.get("status") == "truncated":
        if (manifest.get("termination_reason") not in legacy.scoring.HEADLESS_TRUNCATIONS
                or final.get("shutdown_verified") is not True or final.get("declared_stop") is not True
                or manifest.get("returncode") not in (-15, -9)
                or any(manifest.get(key) != dict.fromkeys(COLORS) for key in ("outcomes_by_player", "rewards_by_player"))):
            raise ReplayError("Unverified or scored cutoff")
    else:
        raise ReplayError("Failed episodes cannot receive a verified contextual replay label")
    return final


def cross_check_wire(episode, manifest, tape):
    """Bind the verifier's regenerated wire to exactly what each player received."""
    profile = contract.profile_for(manifest["environment"])
    header = tape[0] if tape else {}
    army_information = declared_army_information(manifest, profile.environment)
    if (declared_army_information(header, profile.environment) != army_information
            or manifest.get("launch", {}).get("environment", {}).get("EXPLORER_RL_ARMY_INFORMATION") != army_information):
        raise ReplayError("Native replay header or launch differs from the army information declaration")
    expected = {"schema": SCHEMA, "episode": manifest["episode"], "interface": profile.environment,
                "protocol_version": profile.protocol, "history_schema": profile.history_schema,
                "action_schema": profile.action_schema}
    if (type(header.get("protocol_version")) is not int
            or any(header.get(key) != value for key, value in expected.items())):
        raise ReplayError("Contextual capture header does not match the episode")
    for seq, row in enumerate(tape[1:]):
        if type(row.get("seq")) is not int or row["seq"] != seq:
            raise ReplayError("Contextual capture sequence has a gap or reordering")
    counts = Counter(row.get("kind") for row in tape[1:])
    players = {}
    for player, color in enumerate(COLORS):
        bridge, tail = records(episode / f"agents/{color}/bridge.jsonl")
        received = [row["message"] for row in bridge if row.get("direction") == "engine"]
        messages = [row["message"] for row in tape[1:] if row.get("kind") == "message" and row.get("player") == player]
        if messages != received:
            raise ReplayError(f"{color}: captured native messages differ from the delivered bridge")
        delivered = []
        for row in bridge:
            if row.get("direction") != "policy":
                continue
            message = dict(row["message"])
            sent = message.pop("delivered")
            outside = message.pop("outside_learner_prefix")
            if type(sent) is not bool or type(outside) is not bool:
                raise ReplayError("Invalid bridge delivery declaration")
            if sent:
                delivered.append(message)
        consumed = [row["response"] for row in tape[1:] if row.get("kind") == "response" and row.get("player") == player]
        # A final successfully sent intention may lose a race with terminal
        # supersession. It is not a consumed action and earns no target.
        if consumed != delivered[:len(consumed)] or len(delivered) - len(consumed) not in (0, 1):
            raise ReplayError(f"{color}: native response receipts differ from submitted actions")
        if manifest["status"] == "truncated" and len(consumed) != len(delivered):
            raise ReplayError(f"{color}: cutoff has an unconsumed submitted action")
        players[color] = {"messages": len(received), "responses": len(consumed),
                          "submitted_without_receipt": len(delivered) - len(consumed), "verified": True}
    return {"players": players, "transactions": counts["enter"], "messages": counts["message"],
            "responses": counts["response"], "requests": counts["request"]}


def run_native(binary, environment, log, tape, report, *, cutoff, timeout):
    command = [str(binary), "--log", str(log), "--contextual", str(tape), "--report", str(report)]
    if cutoff:
        command.append("--prefix")
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
    started = time.monotonic()
    with report.with_suffix(".log").open("xb") as console:
        result = subprocess.run(command, cwd=binary.parent, env={**inherited, **environment},
            stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, timeout=timeout)
    native = evidence.strict_json(report.read_bytes())
    if native.get("schema") != REPORT_SCHEMA or result.returncode not in (0, 1, 3):
        raise ReplayError(f"Unexpected contextual verifier response: {result.returncode}")
    native["returncode"], native["seconds"] = result.returncode, round(time.monotonic() - started, 3)
    return native


def validate_native_report(native, wire, final, *, cutoff):
    """Admit only complete coverage and the exact regenerated terminal receipt."""
    summary = native.get("summary", {})
    if (native.get("verified") is not True or type(native.get("returncode")) is not int
            or native["returncode"] != 0 or summary.get("server_verified") is not True):
        raise ReplayError("Native replay did not verify successfully")
    loaded, checked = summary.get("loaded", {}), summary.get("checked", {})
    keys = {"transactions", "messages", "responses", "requests", "client_packets"}
    if (type(loaded) is not dict or type(checked) is not dict or set(loaded) != keys or set(checked) != keys
            or any(type(value) is not int or value < 0 for value in (*loaded.values(), *checked.values()))
            or loaded != checked or summary.get("unchecked") != dict.fromkeys(keys, 0)):
        raise ReplayError("Native replay left captured evidence unchecked")
    for key in keys - {"client_packets"}:
        if loaded[key] != wire[key]:
            raise ReplayError(f"Native replay coverage mismatch: {key}")
    boundary = summary.get("boundary", {})
    if (boundary.get("kind") != ("cutoff" if cutoff else "natural")
            or type(boundary.get("client_packets")) is not int
            or boundary["client_packets"] != loaded["client_packets"]
            or summary.get("captured_prefix_only") is not cutoff):
        raise ReplayError("Native replay boundary mismatch")
    if cutoff:
        if summary.get("terminal_barrier", "missing") is not None or boundary.get("barrier_id") is not None:
            raise ReplayError("Cutoff replay cannot claim a natural terminal barrier")
    else:
        barrier = final["terminal_barrier"]["client"]
        # JSON equality also distinguishes booleans/floats from integer fields.
        # Pairwise self-consistency of saved receipts alone cannot bind the
        # request, recipient and outcomes to the actual regenerated barrier.
        if json.dumps(summary.get("terminal_barrier"), sort_keys=True) != json.dumps(barrier, sort_keys=True):
            raise ReplayError("Regenerated terminal barrier differs from the saved client receipt")
        if (summary.get("open_final_transaction") is not False
                or type(boundary.get("barrier_id")) is not int or boundary["barrier_id"] != barrier["barrier_id"]
                or boundary["client_packets"] != barrier["client_packets"]
                or type(summary.get("regenerated_server_packets_after_prefix")) is not int
                or summary["regenerated_server_packets_after_prefix"] != 0):
            raise ReplayError("Natural replay does not reach the recorded callback barrier")


def verify_episode(episode, *, binary, data, output=None, timeout=300):
    episode, binary, data = (Path(p).expanduser().resolve() for p in (episode, binary, data))
    manifest_path = episode / "manifest.json"
    manifest = evidence.strict_json(manifest_path.read_bytes())
    if manifest.get("replay", {}).get("schema") != SCHEMA:
        raise ReplayError("Contextual replay is unsupported for this historical recording; fresh capture required")
    directory = Path(output or episode / "contextual-verification").expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": REPORT_SCHEMA, "kind": "contextual-native-replay", "verified": False,
              "episode": manifest.get("episode"), "created_at": datetime.now(timezone.utc).isoformat(),
              "inputs": {"manifest": evidence.file_identity(manifest_path),
                         "verifier_sources": {str(path.relative_to(ROOT)): evidence.file_identity(path)
                             for path in (Path(__file__), Path(evidence.__file__), Path(legacy.__file__),
                                          ROOT / "experiments/headless/contextual_replay_history.py",
                                          ROOT / "experiments/headless/army_replay_mutations.py",
                                          ROOT / "experiments/headless/dialog_replay_mutations.py",
                                          ROOT / "validation/probes/audit_contextual_histories.py",
                                          ROOT / "experiments/headless/contextual_protocol.py",
                                          Path(contract.__file__)) if path.is_file()}}, "training_performed": False}
    try:
        final = bound_evidence(episode, manifest, binary)
        cutoff = manifest["status"] == "truncated"
        tape, tape_tail = records(episode / "engine/contextual-replay.jsonl", cutoff=cutoff)
        wire = cross_check_wire(episode, manifest, tape)
        report["wire"] = wire
        server, server_tail = records(episode / "engine/replay.jsonl", cutoff=cutoff)
        server, unpaired = legacy.complete_replay_records(server, cut_short=cutoff)
        if server[0].get("seed") != manifest.get("seed"):
            raise ReplayError("Replay seed differs from the episode")
        legacy.write_jsonl(directory / "replay.jsonl", server)
        legacy.write_jsonl(directory / "contextual-replay.jsonl", tape)
        report["prefix"] = {"server_partial_bytes": server_tail, "server_unpaired_inputs": unpaired,
                            "contextual_partial_bytes": tape_tail}
        environment, game_map = legacy.replay_profile(directory, episode, data, manifest["launch"]["requested_settings"])
        if evidence.file_identity(game_map)["sha256"] != manifest["identity"].get("map_sha256"):
            raise ReplayError("Copied replay map differs from the recorded map")
        from experiments.headless.contextual_replay_history import verify_histories
        report["histories"] = verify_histories(episode, manifest, final)
        native = run_native(binary, environment, directory / "replay.jsonl", directory / "contextual-replay.jsonl",
                            directory / "native-report.json", cutoff=cutoff, timeout=timeout)
        report["native"] = native
        validate_native_report(native, wire, final, cutoff=cutoff)
        report["verified"] = True
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(directory / "profile", ignore_errors=True)
    legacy.scoring.write_json(directory / "contextual-replay-verification.json", report)
    return report


def _hero_recruitment_mutations(tape, add):
    """Alter only recorded v5/v6 tavern fields and witnessed hire effects.

    These are diagnostic counterexamples, never constructed positive engine
    evidence. The native verifier must first pass the original recording.
    """
    header = tape[0] if tape else {}
    profile = (contract.MANAGEMENT_PROFILE if header.get("interface") == contract.MANAGEMENT_ENVIRONMENT
               else contract.HEROES_PROFILE)
    expected = {"schema": SCHEMA, "interface": profile.environment, "protocol_version": profile.protocol,
                "history_schema": profile.history_schema, "action_schema": profile.action_schema}
    if (any(header.get(key) != value for key, value in expected.items())
            or type(header.get("protocol_version")) is not int):
        return
    selected = set()

    def change(name, index, path, value):
        if name in selected:
            return
        def mutate(rows, i):
            target = rows[i]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = deepcopy(value)
        add(name, index, mutate)
        selected.add(name)

    observations = []
    for index, row in enumerate(tape):
        if row.get("kind") != "message":
            continue
        for record_index, record in enumerate(row["message"].get("records", [])):
            observation = record.get("observation")
            if observation is None:
                continue
            observations.append((index, record_index, row["player"], observation))
            window = observation.get("window") or {}
            if window.get("context", {}).get("panel") != "tavern":
                continue
            tavern = (window.get("town_panel") or {}).get("tavern")
            if not tavern:
                continue
            prefix = ("message", "records", record_index, "observation", "window", "town_panel", "tavern")
            change("tavern_gold_price", index, (*prefix, "gold_cost"), tavern["gold_cost"] + 1)
            offers = tavern["offers"]
            if offers:
                card = offers[0]
                change("tavern_offer_portrait", index, (*prefix, "offers", 0, "portrait"), card["portrait"] + 1)
                name = next((card["name_text_id"] for card in offers if card["name_text_id"] != offers[0]["name_text_id"]),
                            "hero.core.edric.name" if card["name_text_id"] == "hero.core.orrin.name" else "hero.core.orrin.name")
                change("tavern_offer_name", index, (*prefix, "offers", 0, "name_text_id"), name)
                alternative = next((card["ref"] for card in offers if card["ref"] != tavern["selected_offer_ref"]), None)
                if alternative is not None:
                    change("tavern_selected_offer", index, (*prefix, "selected_offer_ref"), alternative)
            if tavern["selected_detail"] is not None:
                for field, label in (("level", "level"), ("hero_class", "class"), ("artifact_count", "artifact_count")):
                    change("tavern_selected_" + label, index, (*prefix, "selected_detail", field), tavern["selected_detail"][field] + 1)

    for index, row in enumerate(tape):
        if row.get("kind") != "response":
            continue
        action = row["response"]["action"]
        if action["kind"] == "open_panel" and action.get("panel") == "tavern":
            change("tavern_open_navigation", index, ("response", "action", "panel"), "hall")
        if action["kind"] not in ("select_hero", "hire_hero"):
            continue
        prior = next((entry for entry in reversed(observations) if entry[0] < index and entry[2] == row["player"]), None)
        if prior is None:
            continue
        before = prior[3]
        tavern = (((before.get("window") or {}).get("town_panel") or {}).get("tavern"))
        if not tavern:
            continue
        alternative = next((card["ref"] for card in tavern["offers"] if card["ref"] != action["selection"]),
                           action["selection"] + "-changed")
        change("tavern_hero_selection" if action["kind"] == "select_hero" else "tavern_hire_selection",
               index, ("response", "action", "selection"), alternative)
        if action["kind"] != "hire_hero":
            continue
        decision = row["response"].get("decision_id")
        if type(decision) is not int:
            continue
        after = None
        result = None
        acknowledged = None
        # A new hero elsewhere/later is not evidence for this hire. Stop at the
        # matching result or supersession, considering this player's views only.
        for later in range(index + 1, len(tape)):
            message = tape[later]
            if message.get("kind") != "message" or message.get("player") != row["player"]:
                continue
            for record_index, record in enumerate(message["message"].get("records", [])):
                if record.get("observation") is not None:
                    after = (later, record_index, record["observation"])
                if record.get("decision_id") == decision:
                    if record.get("kind") == "acknowledged":
                        acknowledged = (later, record_index, record)
                    if record.get("kind") in ("result", "superseded"):
                        result = record
                        break
            if result is not None:
                break
        if (result is None or result.get("result", {}).get("status") != "hired"
                or result["result"].get("accepted") is not True or after is None):
            continue
        if acknowledged is not None and acknowledged[2].get("accepted") is True:
            change("hire_acknowledgement", acknowledged[0],
                   ("message", "records", acknowledged[1], "accepted"), False)
        after_index, record_index, observed = after
        prior_refs = {hero["handle"] for hero in before["heroes"]}
        added = [(hero_index, hero) for hero_index, hero in enumerate(observed["heroes"]) if hero["handle"] not in prior_refs]
        if len(added) != 1:
            continue
        hero_index, hero = added[0]
        prefix = ("message", "records", record_index, "observation")
        change("hired_hero_creation", after_index, (*prefix, "heroes"),
               [value for value in observed["heroes"] if value["handle"] != hero["handle"]])
        change("hired_hero_gold_effect", after_index, (*prefix, "resources", 6), observed["resources"][6] + 1)
        # Keep the coordinate structurally in bounds. This is a replay negative,
        # not a movement heuristic or action supplied to a game policy.
        for axis, extent in enumerate(observed["size"][:2]):
            if extent > 1:
                change("hired_hero_position", after_index, (*prefix, "heroes", hero_index, "position", axis),
                       (hero["position"][axis] + 1) % extent)
                break
        window = observed.get("window") or {}
        exchange = ((window.get("town_panel") or {}).get("armies") or {})
        visitor = exchange.get("destination")
        visitor_path = ("window", "town_panel", "armies", "destination")
        if profile.army_management:
            match = next(((i, army) for i, army in enumerate(window.get("armies", []))
                          if army["role"] == "visiting_hero" and army["subject"] == hero["handle"]), None)
            visitor = match[1] if match else None
            if match: visitor_path = ("window", "armies", match[0])
        if visitor is not None and visitor.get("subject") == hero["handle"]:
            stack = next(((slot_index, slot) for slot_index, slot in enumerate(visitor["slots"]) if slot["count"]), None)
            if stack is not None:
                slot_index, slot = stack
                change("hired_hero_army_count", after_index,
                       (*prefix, *visitor_path, "slots", slot_index, "count"),
                       slot["count"] + 1)


def tamper_cases(tape):
    """Fresh diagnostic copies; hashes are recomputed before semantic checks."""
    cases = []

    def add(name, index, mutate):
        changed = deepcopy(tape)
        mutate(changed, index)
        for seq, row in enumerate(changed[1:]):
            row["seq"] = seq
        cases.append((name, changed))

    messages = [i for i, row in enumerate(tape) if row.get("kind") == "message"]
    requests = [i for i, row in enumerate(tape) if row.get("kind") == "request"]
    responses = [i for i, row in enumerate(tape) if row.get("kind") == "response"]
    if messages:
        add("omitted_message", messages[len(messages) // 2], lambda rows, i: rows.pop(i))
        add("wrong_recipient", messages[0], lambda rows, i: rows[i].update(player=1 - rows[i]["player"]))
    if len(messages) > 1:
        first, second = messages[:2]
        def swap(rows, unused):
            rows[first], rows[second] = rows[second], rows[first]
        add("reordered_messages", first, swap)
    if requests:
        add("omitted_native_request", requests[0], lambda rows, i: rows.pop(i))
        add("request_identity", requests[0], lambda rows, i: rows[i].update(request_id=rows[i]["request_id"] + 1))
        def corrupt_pack(rows, i):
            raw = rows[i]["pack"]
            rows[i]["pack"] = ("0" if raw[0] != "0" else "1") + raw[1:]
        add("native_command_bytes", requests[0], corrupt_pack)
        add("request_source_spoof", requests[0], lambda rows, i: rows[i].update(
            source="automatic" if rows[i]["source"] == "adapter" else "adapter"))
    selected = set()
    for index in messages:
        for record_index, record in enumerate(tape[index]["message"].get("records", [])):
            observation = record.get("observation")
            if observation and observation.get("tiles") and "road" not in selected:
                def road(rows, i, r=record_index):
                    rows[i]["message"]["records"][r]["observation"]["tiles"][0]["road"] += 1
                add("observed_road", index, road)
                selected.add("road")
            if observation and observation.get("window") and "window" not in selected:
                def window(rows, i, r=record_index):
                    rows[i]["message"]["records"][r]["observation"]["window"] = None
                add("open_window_state", index, window)
                selected.add("window")
            if record.get("event", {}).get("kind") == "notice" and "notice" not in selected:
                def notice(rows, i, r=record_index):
                    rows[i]["message"]["records"].pop(r)
                add("omitted_notice", index, notice)
                selected.add("notice")
    for index in responses:
        action = tape[index]["response"]["action"]
        kind = action["kind"]
        if kind == "move_to" and "destination" not in selected:
            def destination(rows, i):
                rows[i]["response"]["action"]["x"] += 1
            add("destination_coordinate", index, destination)
            selected.add("destination")
        if kind == "recruit" and "recruit" not in selected:
            def recruitment(rows, i):
                rows[i]["response"]["action"]["count"] += 1
            add("exact_recruitment_count", index, recruitment)
            selected.add("recruit")
        if kind == "transfer_stack" and "transfer" not in selected:
            # Prefer another actually displayed slot. The changed count remains
            # supported when the original count exceeds one, and an alternative
            # destination may also be valid: replay must detect the different
            # native command/effect, not just reject malformed identifiers.
            armies = None
            for prior in reversed(tape[:index]):
                if prior.get("kind") != "message" or prior.get("player") != tape[index]["player"]:
                    continue
                observation = next((record["observation"] for record in
                                    reversed(prior["message"].get("records", [])) if record.get("observation")), None)
                if observation is not None:
                    armies = ((observation.get("window") or {}).get("town_panel") or {}).get("armies")
                    break
            source_ref, destination_ref = action["selection"] + "-changed", action["destination"] + "-changed"
            if armies and armies.get("destination"):
                source = next((slot for slot in armies["source"]["slots"] if slot["ref"] == action["selection"]), None)
                source_ref = next((slot["ref"] for slot in armies["source"]["slots"]
                                   if slot["ref"] != action["selection"]), source_ref)
                destination_ref = next((slot["ref"] for slot in armies["destination"]["slots"]
                                        if slot["ref"] != action["destination"] and source
                                        and slot["creature"] in (None, source["creature"])), destination_ref)
            add("exact_transfer_count", index, lambda rows, i: rows[i]["response"]["action"].update(
                count=action["count"] - 1 if action["count"] > 1 else action["count"] + 1))
            add("explicit_transfer_source_slot", index, lambda rows, i: rows[i]["response"]["action"].update(
                selection=source_ref))
            add("explicit_transfer_destination_slot", index, lambda rows, i: rows[i]["response"]["action"].update(
                destination=destination_ref))
            selected.add("transfer")
        if kind == "construct" and "construct" not in selected:
            def construction(rows, i):
                rows[i]["response"]["action"]["selection"] += "-changed"
            add("construction_selection", index, construction)
            selected.add("construct")
        if kind in ("open_hero", "open_town", "close", "back") and "navigation" not in selected:
            def navigation(rows, i):
                rows[i]["response"]["action"]["kind"] = "end_turn"
            add("window_navigation", index, navigation)
            selected.add("navigation")
    _hero_recruitment_mutations(tape, add)
    army_management_mutations(tape, add)
    adventure_dialogs_mutations(tape, add)
    closes = [i for i, row in enumerate(tape) if row.get("kind") == "close"]
    if closes:
        add("missing_natural_close", closes[-1], lambda rows, i: rows.pop(i))
    return cases


def tamper_episode(episode, *, binary, data, output, timeout=300):
    """Challenge native semantics directly after validating the original binding.

    Each altered tape is given a fresh SHA-256. The outer file-hash check is
    deliberately not the rejection mechanism in these negative tests.
    """
    episode, binary, data, directory = (Path(p).expanduser().resolve() for p in (episode, binary, data, output))
    manifest = evidence.strict_json((episode / "manifest.json").read_bytes())
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": "heroes-contextual-replay-tamper-v1", "cases": [], "passed": False,
              "scope": "Semantic native replay of rehashed diagnostic mutations; original artifacts preserved"}
    source_names = ("manifest.json", "contextual-finalization.json", *evidence.REPLAY_FILES)
    original_inputs = {name: evidence.file_identity(evidence.inside(episode, name)) for name in source_names}
    try:
        # A verifier that fails on every input is not a successful tamper test.
        # Require the complete, unmodified episode (including reconstructed
        # histories) to pass before interpreting any negative result.
        positive = verify_episode(episode, binary=binary, data=data, output=directory / "positive-control", timeout=timeout)
        positive_path = directory / "positive-control/contextual-replay-verification.json"
        if evidence.strict_json(positive_path.read_bytes()) != positive:
            raise ReplayError("Positive-control report differs from its saved artifact")
        report["positive_control"] = {"verified": positive.get("verified") is True,
            "path": str(positive_path.relative_to(directory)), "identity": evidence.file_identity(positive_path),
            "inputs": original_inputs}
        if positive.get("verified") is not True:
            report["error"] = "Unmodified episode failed its positive control; negative tests were not run"
            legacy.scoring.write_json(directory / "tamper-report.json", report)
            return report
        if original_inputs != {name: evidence.file_identity(evidence.inside(episode, name)) for name in source_names}:
            raise ReplayError("Original replay inputs changed during the positive control")
        bound_evidence(episode, manifest, binary)
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        legacy.scoring.write_json(directory / "tamper-report.json", report)
        return report
    cutoff = manifest["status"] == "truncated"
    tape, _ = records(episode / "engine/contextual-replay.jsonl", cutoff=cutoff)
    server, _ = records(episode / "engine/replay.jsonl", cutoff=cutoff)
    server, _ = legacy.complete_replay_records(server, cut_short=cutoff)
    for name, changed in tamper_cases(tape):
        case = directory / name
        case.mkdir()
        legacy.write_jsonl(case / "replay.jsonl", server)
        legacy.write_jsonl(case / "contextual-replay.jsonl", changed)
        environment, _ = legacy.replay_profile(case, episode, data, manifest["launch"]["requested_settings"])
        row = {"name": name, "input": evidence.file_identity(case / "contextual-replay.jsonl"), "detected": False}
        try:
            native = run_native(binary, environment, case / "replay.jsonl", case / "contextual-replay.jsonl",
                                case / "native-report.json", cutoff=cutoff, timeout=timeout)
            row["native"] = native
            row["detected"] = native.get("verified") is False and native.get("returncode") in (1, 3)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            shutil.rmtree(case / "profile", ignore_errors=True)
        report["cases"].append(row)
        legacy.scoring.write_json(directory / "tamper-report.json", report)
    report["passed"] = bool(report["cases"]) and all(row["detected"] for row in report["cases"])
    legacy.scoring.write_json(directory / "tamper-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--binary", type=Path, default=legacy.DEFAULT_REPLAY_BINARY)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--tamper", action="store_true", help="Run semantic negative checks on rehashed copies")
    args = parser.parse_args(argv)
    try:
        if args.tamper:
            if args.output is None:
                parser.error("--tamper requires a new --output directory")
            result = tamper_episode(args.episode, binary=args.binary, data=args.data, output=args.output, timeout=args.timeout)
        else:
            result = verify_episode(args.episode, binary=args.binary, data=args.data, output=args.output, timeout=args.timeout)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("verified", result.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
