"""Verify captured Nullkiller commands and the one learner's native v7/v8 stream.

The native verifier regenerates engine effects from the recorded server inputs,
then runs ExplorerRL for the learner and a passive opponent. This is exact
captured-command replay, never a claim that Nullkiller is deterministic by seed.
Opponent packets are evidence only, not observations or training examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from experiments.headless import background_check
from experiments.headless import builtin_evidence as builtin
from experiments.headless import contextual_evidence as evidence
from experiments.headless import contextual_replay as paired
from experiments.headless import embedded_evidence as embedded
from experiments.headless import replay as legacy
from experiments.learner import contextual as contract

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-contextual-builtin-replay-v1"
REPORT_SCHEMA = "heroes-contextual-builtin-replay-verification-v1"
DIAGNOSTIC_SCHEMA = "heroes-contextual-builtin-replay-diagnostic-v1"
COLORS = ("red", "blue")
ReplayError = legacy.ReplayError


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def controllers(player):
    require(type(player) is int and player in (0, 1), "Invalid learner controller")
    return {"schema": "heroes-contextual-controllers-v1", "explorer_players": [player],
            "native_opponents": [{"player": 1 - player, "module": "Nullkiller"}]}


def cross_check_wire(episode, manifest, tape, server):
    """Bind all requests to server inputs or explicit terminal-unapplied accounting."""
    player = manifest["learner_player"]
    expected_controllers = controllers(player)
    require(equal(manifest.get("controllers"), expected_controllers), "Manifest controllers differ")
    profile = builtin.manifest_profile(manifest)
    header = {"kind": "init", "schema": SCHEMA, "episode": manifest["episode"],
        "interface": profile.environment, "protocol_version": profile.protocol,
        "history_schema": profile.history_schema, "action_schema": profile.action_schema,
        "controllers": expected_controllers}
    require(tape and equal(tape[0], header), "Mixed capture header differs from manifest/controllers")
    require(manifest["environment"] == profile.environment, "Built-in replay interface mismatch")
    requests, request_contexts, messages, responses, counts = [], [], [], [], Counter()
    transaction, closed = None, False
    receipt, close_record = None, None
    for seq, row in enumerate(tape[1:]):
        require(type(row.get("seq")) is int and row["seq"] == seq and not closed, "Mixed tape sequence gap or record after close")
        kind = row.get("kind")
        if kind == "enter":
            require(transaction is None, "Nested native transaction")
            transaction = (row.get("actor"), row.get("operation"))
            receipt = row.get("packet")
            require(transaction[0] in ("client", f"player{player}"), "Unexpected transaction controller")
            counts["transactions"] += 1
            if transaction == ("client", "packet"):
                counts["client_packets"] += 1
        elif kind == "leave":
            require(transaction is not None and transaction == (row.get("actor"), row.get("operation")), "Unmatched transaction leave")
            transaction = None
        elif kind == "close":
            require(transaction is None and type(row.get("barrier_id")) is int and row["barrier_id"] == 1,
                    "Close is not after the native terminal transaction")
            closed = True
            close_record = row
        else:
            require(transaction is not None and kind in ("message", "response", "request", "packet_complete"), "Payload outside a native transaction")
            if kind in ("message", "response"):
                require(type(row.get("player")) is int and row["player"] == player, "Native opponent cannot acquire a learner wire stream")
                (messages if kind == "message" else responses).append(row["message" if kind == "message" else "response"])
                counts["messages" if kind == "message" else "responses"] += 1
            elif kind == "request":
                require(set(row) == {"seq", "kind", "player", "request_id", "pack_type", "pack", "source"}, "Malformed request record")
                require(all(type(row[key]) is int for key in ("player", "request_id", "pack_type"))
                        and row["player"] in (-1, 0, 1) and 0 <= row["request_id"] <= 0xffffffff
                        and 0 < row["pack_type"] <= 0xffff, "Invalid native request identity")
                encoded = row["pack"]
                require(type(encoded) is str and bool(encoded) and len(encoded) % 2 == 0
                        and all(c in "0123456789abcdef" for c in encoded), "Invalid serialized native request")
                source = row["source"]
                if source == "adapter":
                    require(row["player"] == player and transaction[0] == f"player{player}", "Adapter request is outside learner transaction")
                elif source == "native_opponent":
                    require(row["player"] == 1 - player and transaction[0] == "client"
                            and transaction[1] in ("request", "packet"), "Native opponent source/controller mismatch")
                    counts["native_opponent_requests"] += 1
                else:
                    require(source == "automatic" and row["player"] != 1 - player and transaction[0] == "client"
                            and transaction[1] in ("request", "packet"), "Unsupported automatic request source")
                    # The native verifier additionally decodes and enforces the
                    # existing AdvInterfaceReady / MakeAction type whitelist.
                requests.append(row)
                request_contexts.append((transaction, receipt))
                counts["requests"] += 1
    cutoff = manifest["status"] == "truncated"
    require((cutoff and not closed) or (not cutoff and closed and transaction is None), "Native tape boundary differs from episode")
    inputs = server[2::2]
    require(len(server) >= 2 and server[0].get("kind") == "init" and server[1].get("kind") == "start"
            and len(server) % 2 == 0 and len(inputs) <= len(requests) <= len(inputs) + 1,
            "Native request coverage differs from authoritative inputs")
    for index, (request, received, applied) in enumerate(zip(requests, inputs, server[3::2])):
        require(received.get("kind") == "input" and applied.get("kind") == "applied"
                and type(received.get("seq")) is int and received["seq"] == index
                and type(applied.get("seq")) is int and applied["seq"] == index, "Authoritative input/effect order changed")
        require(all(equal(request[left], received.get(right)) for left, right in
                    (("pack", "pack"), ("pack_type", "type"), ("player", "player"), ("request_id", "request"))),
                "Native request bytes/identity/order differ from authoritative server input")
    accounting = request_accounting(manifest, tape, server, requests, request_contexts, close_record)
    bridge, _ = paired.records(Path(episode) / f"agents/{COLORS[player]}/bridge.jsonl")
    received = [row["message"] for row in bridge if row.get("direction") == "engine"]
    require(equal(received, messages), "Learner native messages differ from delivered bridge")
    delivered = []
    for row in bridge:
        if row.get("direction") != "policy":
            continue
        response = dict(row["message"])
        require(response.pop("delivered", None) is True and response.pop("outside_learner_prefix", None) is False,
                "Unverified delivery or invented technical continuation")
        delivered.append(response)
    require(equal(responses, delivered[:len(responses)]) and len(delivered) - len(responses) in (0, 1),
            "Native learner consumption differs from submitted responses")
    require(not cutoff or len(responses) == len(delivered), "Cutoff contains an unconsumed submitted response")
    return {**{key: counts[key] for key in ("transactions", "messages", "responses", "requests", "client_packets", "native_opponent_requests")},
        "controllers": expected_controllers, "learner_player": player,
        "submitted_without_receipt": len(delivered) - len(responses),
        "request_accounting": accounting,
        "request_order_verified": True, "opponent_history": None,
        "scope": "Authoritative command prefix and separately accounted terminal-unapplied attempt; engine effects and receipt positions require independent native replay."}


def request_accounting(manifest, tape, server, requests, contexts, close):
    """Never discard a request or infer that a queued terminal attempt was rejected."""
    inputs = server[2::2]
    final = server[-1] if inputs else {}
    terminal = None
    if final.get("boundary") == "game_end":
        last = inputs[-1]
        previous = server[-3] if len(inputs) > 1 else server[1]
        first, end = previous.get("client_out"), final.get("client_out")
        require(type(first) is int and type(end) is int and 0 <= first < end,
                "Final terminal packet range is invalid")
        terminal = {"input_seq": last["seq"], "player": last["player"], "request_id": last["request"],
                    "pack_type": last["type"], "first_client_packet": first + 1, "last_client_packet": end}
    tail = requests[len(inputs):]
    accounted = []
    if tail:
        require(manifest["status"] == "completed" and terminal is not None and close is not None,
                "Unapplied attempt requires a closed natural terminal")
        outcomes = final.get("terminals")
        require(type(outcomes) is list and len(outcomes) == 2
                and all(type(x) is dict and set(x) == {"player", "victory"}
                        and type(x["player"]) is int and type(x["victory"]) is bool for x in outcomes)
                and {x["player"] for x in outcomes} == {0, 1}
                and {x["victory"] for x in outcomes} == {False, True},
                "Unapplied attempt lacks complementary native terminal outcomes")
        row = tail[0]
        transaction, receipt = contexts[-1]
        require(row["source"] == "native_opponent" and row["player"] == 1 - manifest["learner_player"]
                and row["pack_type"] in (180, 182, 185) and transaction == ("client", "request"),
                "Only terminal native-opponent MoveHero, BuildStructure or EndTurn attempts are supported")
        require(inputs[-1]["request"] < 0xffffffff and row["request_id"] == inputs[-1]["request"] + 1,
                "Terminal-unapplied request ID is not contiguous")
        require(type(receipt) is dict and set(receipt) == {"index", "completed", "phase", "type"}
                and all(type(receipt[k]) is int for k in ("index", "completed", "type"))
                and receipt["phase"] == "callbacks_complete" and receipt["index"] == receipt["completed"]
                and terminal["first_client_packet"] <= receipt["index"] < terminal["last_client_packet"]
                and 0 < receipt["type"] <= 0xffff
                and type(close.get("client_packets")) is int
                and close["client_packets"] == terminal["last_client_packet"],
                "Terminal-unapplied attempt is outside the final request's packet range")
        require(not any(x.get("kind") == "response" and x["seq"] > row["seq"] for x in tape[1:]),
                "Learner response follows the terminal-unapplied attempt")
        # Native replay additionally decodes the packet, checks the exact
        # regenerated receipt/type, terminal state, final ACK and barrier.
        accounted.append({key: row[key] for key in ("seq", "player", "request_id", "pack_type", "pack", "source")}
                         | {"receipt": receipt})
    return {"authoritative_inputs": len(inputs), "terminal_unapplied": accounted, "final_terminal_input": terminal}


def validate_native_report(native, wire, final, *, cutoff):
    require(native.get("schema") == REPORT_SCHEMA, "Wrong mixed native report schema")
    summary = native.get("summary", {})
    require(equal(summary.get("controllers"), wire["controllers"]), "Regenerated controller binding differs")
    loaded, checked, unchecked = (summary.get(key, {}) for key in ("loaded", "checked", "unchecked"))
    keys = {"transactions", "messages", "responses", "requests", "client_packets", "native_opponent_requests"}
    require(all(type(values) is dict and set(values) == keys for values in (loaded, checked, unchecked))
            and all(type(value) is int and value >= 0 for values in (loaded, checked, unchecked) for value in values.values())
            and loaded == checked and unchecked == dict.fromkeys(keys, 0)
            and all(loaded[key] == wire[key] for key in keys), "Native mixed coverage is incomplete")
    # Preserve the existing strict packet-boundary and regenerated barrier
    # admission without relaxing its paired report format.
    conventional = json.loads(json.dumps(native))
    for key in ("loaded", "checked", "unchecked"):
        conventional["summary"][key].pop("native_opponent_requests")
    paired.validate_native_report(conventional, wire, final, cutoff=cutoff)
    accounting = wire.get("request_accounting")
    native_accounting = summary.get("request_accounting")
    if accounting is None and native_accounting is None:
        # Historical zero-tail reports retain their original strict equality.
        authoritative = wire["requests"]
    else:
        require(type(accounting) is dict and equal(accounting, native_accounting)
                and set(accounting) == {"authoritative_inputs", "terminal_unapplied", "final_terminal_input"},
                "Native terminal request accounting differs from the captured wire")
        authoritative, tail = accounting["authoritative_inputs"], accounting["terminal_unapplied"]
        require(type(authoritative) is int and authoritative >= 0 and type(tail) is list
                and len(tail) in (0, 1) and (not cutoff or not tail)
                and authoritative + len(tail) == wire["requests"], "Invalid terminal-unapplied request coverage")
        if tail:
            origin = accounting["final_terminal_input"]
            barrier = final["terminal_barrier"]["client"]
            require(type(origin) is dict and barrier["request"] == {
                "player": origin.get("player"), "id": origin.get("request_id"), "type": origin.get("pack_type")}
                and barrier["client_packets"] == origin.get("last_client_packet"),
                "Terminal-unapplied accounting differs from final barrier origin")
    require(type(summary.get("server", {}).get("inputs")) is int
            and summary["server"]["inputs"] == authoritative, "Server replay did not cover every authoritative input")


def effective_environment(episode, manifest, final):
    """Recheck actual Easy/routing/fog flags, not just requested settings."""
    path = Path(episode) / "engine/experiment-metadata.json"
    metadata = evidence.strict_json(path.read_bytes())
    player = manifest["learner_player"]
    require(equal(metadata.get("controllers"), controllers(player)), "Effective engine controllers differ")
    log = (Path(episode) / "logs/process.log").read_text(errors="replace")
    checked = {"metadata": legacy.scoring.validate_experiment_metadata(metadata, player, log, agent_player=player),
               "seed": legacy.scoring.engine_seed_evidence(log, manifest["seed"])}
    require(equal(checked, manifest.get("engine_evidence")) and equal(checked, final.get("engine_evidence")),
            "Finalization effective engine settings differ from actual evidence")
    return {"verified": True, "metadata_file": evidence.file_identity(path), "settings": checked}


def verify_identity(manifest, binary, *, diagnostic_verifier_compatibility=False):
    """Only an explicit supplemental diagnostic may change verifier code/binary."""
    require(type(diagnostic_verifier_compatibility) is bool, "Diagnostic mode must be explicit")
    identity = manifest["identity"]
    runtime = identity.get("native_runtime_sha256", {})
    current_runtime = evidence.runtime_identity(binary.parent)
    require(type(runtime) is dict and set(runtime) == set(current_runtime), "Captured native runtime inventory changed")
    require(all(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
                for sha in runtime.values()), "Invalid captured native runtime digest")
    runtime_changes = {name: {"captured": runtime[name], "current": current_runtime[name]}
                       for name in runtime if runtime[name] != current_runtime[name]}
    require((not runtime_changes or diagnostic_verifier_compatibility and set(runtime_changes) <= {"vcmireplay"})
            and current_runtime.get("vcmireplay") == evidence.file_identity(binary)["sha256"],
            "Captured native runtime changed outside the permitted verifier")
    require(identity.get("binary_sha256") == evidence.file_identity(binary.parent / "vcmiclient")["sha256"], "Captured client changed")
    for label, key in (("ExplorerRL", "explorer_rl_sha256"), ("Nullkiller", None)):
        sha = next((value for name, value in runtime.items() if name in (f"AI/lib{label}.dylib", f"AI/lib{label}.so")), None)
        require(sha and sha == (identity.get(key) if key else manifest["opponent_identity"]["binary_sha256"]), f"{label} binary identity differs")
    recipe, current_recipe = identity.get("engine_recipe"), legacy.scoring.engine_recipe_digests(ROOT)
    require(type(recipe) is dict and all(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
                and type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
                for name, sha in recipe.items()), "Invalid captured native source recipe")
    recipe_changes = {name: {"captured": recipe.get(name), "current": current_recipe.get(name)}
                      for name in recipe.keys() | current_recipe.keys() if recipe.get(name) != current_recipe.get(name)}
    require(not recipe_changes or diagnostic_verifier_compatibility
            and all(name.startswith("engine/ExperimentReplay/") for name in recipe_changes),
            "Canonical nonverifier native source changed")
    bridge = {name: sha for name, sha in recipe.items() if name.startswith("engine/ExplorerRL/")}
    require(identity.get("bridge_source_sha256") == hashlib.sha256(
        json.dumps(bridge, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "Captured adapter recipe digest differs")
    expected_sources = evidence.ADAPTER_SOURCES | {"experiments/headless/builtin_env.py", "experiments/headless/builtin_evidence.py"}
    backend = None
    if embedded.is_embedded(identity):
        # The in-process backend runs the same rules from its extension module.
        expected_sources |= embedded.ADAPTER_SOURCES
        backend = embedded.verify_backend(identity, binary.parent, ROOT)
    require(set(identity.get("contextual_python_sha256", {})) == expected_sources, "Incomplete mixed adapter source inventory")
    for name, sha in identity["contextual_python_sha256"].items():
        require(evidence.file_identity(evidence.inside(ROOT, name))["sha256"] == sha, "Adapter source changed: " + name)
    lock = evidence.strict_json((ROOT / "dependencies/vcmi.json").read_bytes())
    require(identity.get("engine_revision") == lock["commit"] and identity.get("engine", {}).get("commit") == lock["commit"], "Pinned engine identity mismatch")
    result = {"mode": "diagnostic_verifier_only" if diagnostic_verifier_compatibility else "exact_capture",
              "runtime_changes": runtime_changes, "recipe_changes": recipe_changes,
              "nonverifier_runtime_unchanged": True, "client_unchanged": True,
              "adapter_sources_unchanged": True, "nonverifier_recipe_unchanged": True}
    if backend is not None:
        result["embedded_backend"] = backend
    return result


def bound_evidence(episode, manifest, binary, *, diagnostic_verifier_compatibility=False, identity_report=None):
    from experiments.headless import builtin_evidence as builtin
    in_process = embedded.is_embedded(manifest.get("identity"))
    final = builtin.read_finalization(episode, manifest)
    require(manifest.get("kind") == builtin.MANIFEST_KIND and manifest.get("replay", {}).get("schema") == SCHEMA,
            "Unsupported built-in recording")
    recorded = manifest["replay"]
    require(recorded.get("recorded") is True and not recorded.get("missing")
            and equal(builtin.capture_files(episode), recorded) and equal(final.get("native_capture"), recorded),
            "Native mixed capture changed after finalization")
    identity = manifest["identity"]
    require(identity.get("captured_before_launch") is True and final.get("identity") == identity,
            "Missing prelaunch native identity")
    checked = verify_identity(manifest, binary, diagnostic_verifier_compatibility=diagnostic_verifier_compatibility)
    if identity_report is not None:
        identity_report.update(checked)
    require(evidence.file_identity(episode / "logs/process.log")["sha256"] == final.get("process_log_sha256"), "Process log changed")
    learner = manifest["learner_player"]
    color = COLORS[learner]
    require(final.get("stream_complete") is True and final.get("engine_evidence_error") is None
            and final.get("terminal_barrier_error") is None
            and final.get("returncode") == manifest.get("returncode"), "Unverified mixed finalization")
    require(set(final.get("raw_bridge_sha256", {})) == {color}
            and evidence.file_identity(episode / f"agents/{color}/bridge.jsonl")["sha256"] == final["raw_bridge_sha256"][color], "Learner bridge changed")
    if manifest["status"] == "completed":
        outcomes = [manifest["outcomes_by_player"][c] for c in COLORS]
        barrier = builtin.terminal_barrier(episode, episode=manifest["episode"], interface=manifest["environment"],
                                           outcomes=outcomes, learner_player=learner)
        evidence.terminal_capture_close(episode, barrier)
        require(equal(barrier, final["terminal_barrier"]) and manifest["returncode"] == 0
                and manifest["termination_reason"] == "natural_terminal"
                and final["socket_outcome"] == outcomes[learner]
                and equal(final["engine_verified_outcomes"], manifest["outcomes_by_player"])
                and equal(manifest["rewards_by_player"], {c: 1 if o == "win" else -1 for c, o in zip(COLORS, outcomes)})
                and (not in_process or embedded.natural_closed(final)),
                "Natural labels differ from actual native barrier")
    else:
        require(manifest["status"] == "truncated" and manifest["termination_reason"] in legacy.scoring.HEADLESS_TRUNCATIONS
                and final.get("declared_stop") is True and final.get("shutdown_verified") is True
                and (embedded.truncation_closed(final) and manifest["returncode"] is None if in_process
                     else manifest["returncode"] in (-15, -9)) and final.get("socket_outcome") is None
                and final.get("engine_verified_outcomes") is None and final.get("terminal_barrier") is None
                and all(manifest.get(key) == dict.fromkeys(COLORS) for key in ("outcomes_by_player", "rewards_by_player")),
                "Failed, unverified or scored cutoff cannot be replay-verified")
    return final


def run_native(binary, environment, log, tape, report, *, cutoff, timeout):
    command = [str(binary), "--log", str(log), "--contextual", str(tape), "--report", str(report)]
    if cutoff:
        command.append("--prefix")
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
    started = time.monotonic()
    with report.with_suffix(".log").open("xb") as console:
        completed = subprocess.run(command, cwd=binary.parent, env={**inherited, **environment},
            stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, timeout=timeout)
    native = evidence.strict_json(report.read_bytes())
    require(native.get("schema") == REPORT_SCHEMA and completed.returncode in (0, 1, 3), "Unexpected mixed native response")
    native.update(returncode=completed.returncode, seconds=round(time.monotonic() - started, 3))
    return native


def diagnostic_inputs(episode, manifest):
    """Hash the original failed admission and captured artifacts, without rewriting them."""
    candidates = [name for name in ("builtin-verification/builtin-replay-verification.json",
                  "training-verification/builtin-replay-verification.json") if (episode / name).is_file()]
    require(len(candidates) == 1, "Supplemental diagnosis requires one unambiguous original admission report")
    original = candidates[0]
    result = evidence.strict_json(evidence.inside(episode, original).read_bytes())
    require(result.get("schema") == REPORT_SCHEMA and result.get("verified") is False
            and equal(result.get("inputs", {}).get("manifest"), evidence.file_identity(episode / "manifest.json")),
            "Supplemental diagnosis requires the preserved failed original admission")
    color = COLORS[manifest["learner_player"]]
    profile = builtin.manifest_profile(manifest)
    names = {"manifest.json", "builtin-finalization.json", original, "logs/process.log", "engine/experiment-metadata.json",
             f"agents/{color}/bridge.jsonl", f"agents/{color}/{profile.history_filename}", f"agents/{color}/live-{profile.history_filename}",
             *manifest["replay"]["files"]}
    return {name: evidence.file_identity(evidence.inside(episode, name)) for name in sorted(names)}


def _started(function, *args):
    """Run ``function`` on a daemon thread; the returned call waits, then returns or raises its result."""
    result = {}
    def run():
        try:
            result["value"] = function(*args)
        except BaseException as error:
            result["error"] = error
    thread = threading.Thread(target=run, name="history-verification", daemon=True)
    thread.start()
    def wait():
        thread.join()  # interruptible: the caller's alarm handler still runs here
        if "error" in result:
            raise result["error"]
        return result["value"]
    return wait


def verify_episode(episode, *, binary, data, output=None, timeout=300, diagnostic_verifier_compatibility=False):
    episode, binary, data = (Path(p).expanduser().resolve() for p in (episode, binary, data))
    manifest = evidence.strict_json((episode / "manifest.json").read_bytes())
    require(manifest.get("replay", {}).get("schema") == SCHEMA, "Fresh mixed capture required; paired/legacy evidence is unsupported")
    require(type(diagnostic_verifier_compatibility) is bool, "Diagnostic mode must be explicit")
    directory = Path(output or episode / "builtin-verification").resolve()
    preserved = None
    if diagnostic_verifier_compatibility:
        require(output is not None and not directory.is_relative_to(episode) and not episode.is_relative_to(directory),
                "Supplemental diagnosis requires an explicit fresh output outside the captured episode")
        preserved = diagnostic_inputs(episode, manifest)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": DIAGNOSTIC_SCHEMA if diagnostic_verifier_compatibility else REPORT_SCHEMA,
        "kind": "builtin-supplemental-verifier-diagnostic" if diagnostic_verifier_compatibility else "builtin-captured-command-replay", "verified": False,
        "episode": manifest.get("episode"), "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"manifest": evidence.file_identity(episode / "manifest.json"),
            "verifier_sources": {str(path.relative_to(ROOT)): evidence.file_identity(path) for path in
                (Path(__file__), ROOT / "experiments/headless/builtin_replay_history.py", ROOT / "experiments/headless/builtin_evidence.py",
                 Path(embedded.__file__), Path(paired.__file__), Path(evidence.__file__), Path(legacy.__file__), Path(contract.__file__))}},
        "training_performed": False, "training_samples": 0, "optimization_eligible": False, "opponent_policy_reexecuted": False,
        "scope": "Recorded native opponent commands, independently regenerated engine effects, exact learner adapter/wire/history."}
    if diagnostic_verifier_compatibility:
        report.update(diagnostic_verified=False, original_cohort_admission=False, original_cohort_result_unchanged=False,
                      outcomes_by_player=dict.fromkeys(COLORS), rewards_by_player=dict.fromkeys(COLORS))
        report["inputs"]["preserved_original_files"] = preserved
        report["scope"] += " Supplemental diagnosis only: never changes the stopped cohort or admits optimization samples."
    try:
        report["identity_check"] = {}
        final = bound_evidence(episode, manifest, binary,
            diagnostic_verifier_compatibility=diagnostic_verifier_compatibility, identity_report=report["identity_check"])
        report["effective_engine"] = effective_environment(episode, manifest, final)
        cutoff = manifest["status"] == "truncated"
        tape, tape_tail = paired.records(episode / "engine/contextual-replay.jsonl", cutoff=cutoff)
        server, server_tail = paired.records(episode / "engine/replay.jsonl", cutoff=cutoff)
        server, unpaired = legacy.complete_replay_records(server, cut_short=cutoff)
        require(server[0].get("seed") == manifest.get("seed") and server[0].get("no_cheats") == "1", "Authoritative seed or no-cheats setting differs")
        report["wire"] = cross_check_wire(episode, manifest, tape, server)
        report["native_opponent_requests"] = report["wire"]["native_opponent_requests"]
        legacy.write_jsonl(directory / "replay.jsonl", server)
        legacy.write_jsonl(directory / "contextual-replay.jsonl", tape)
        report["prefix"] = {"server_partial_bytes": server_tail, "server_unpaired_inputs": unpaired, "contextual_partial_bytes": tape_tail}
        environment, game_map = legacy.replay_profile(directory, episode, data, manifest["launch"]["requested_settings"])
        require(evidence.file_identity(game_map)["sha256"] == manifest["identity"].get("map_sha256"), "Replay map differs from recorded map")
        from experiments.headless.builtin_replay_history import verify_history
        if background_check.FORK:
            # The independent Python history check runs while this thread waits on the native
            # verifier, which reads only the files written above. A history error, which the serial
            # order raised first, is still the error raised; no result is recorded before both pass.
            history = _started(verify_history, episode, manifest, final)
        else:
            # Serial fallback: the accepted order, the history check first and no thread.
            checked = verify_history(episode, manifest, final)
            history = lambda: checked
        try:
            native = run_native(binary, environment, directory / "replay.jsonl", directory / "contextual-replay.jsonl",
                directory / "native-report.json", cutoff=cutoff, timeout=timeout)
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            # The caller's wall limit or a stop ends verification now; the history thread cannot
            # be interrupted, so it is abandoned (it only reads) and records nothing.
            raise
        except BaseException:
            report["history"] = history()
            raise
        report["history"] = history()
        report["native"] = native
        validate_native_report(report["native"], report["wire"], final, cutoff=cutoff)
        if diagnostic_verifier_compatibility:
            require(equal(preserved, diagnostic_inputs(episode, manifest)), "Original evidence changed during supplemental diagnosis")
            report["diagnostic_verified"] = True
        else:
            report["verified"] = True
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(directory / "profile", ignore_errors=True)
        if preserved is not None:
            try:
                current = {name: evidence.file_identity(evidence.inside(episode, name)) for name in preserved}
                report["original_cohort_result_unchanged"] = equal(current, preserved)
                require(report["original_cohort_result_unchanged"], "Original evidence changed during supplemental diagnosis")
            except (ValueError, OSError) as error:
                report["diagnostic_verified"] = False
                report["error"] = f"{type(error).__name__}: {error}"
    name = "builtin-replay-diagnostic.json" if diagnostic_verifier_compatibility else "builtin-replay-verification.json"
    legacy.scoring.write_json(directory / name, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--diagnostic-verifier-compatibility", action="store_true",
                        help="Supplemental replay only: permit verifier-only changes; never admit the original cohort")
    args = parser.parse_args()
    result = verify_episode(args.episode, binary=args.binary, data=args.data, output=args.output, timeout=args.timeout,
                            diagnostic_verifier_compatibility=args.diagnostic_verifier_compatibility)
    print(json.dumps({"verified": result["verified"], "diagnostic_verified": result.get("diagnostic_verified"), "error": result.get("error")}))
    return 0 if result.get("diagnostic_verified", result["verified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
