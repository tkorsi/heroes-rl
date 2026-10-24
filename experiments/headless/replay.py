#!/usr/bin/env python3
"""Verify recorded headless episodes by deterministic replay.

An episode recorded with ``VCMI_EXPERIMENT_REPLAY=1`` leaves three kinds of
engine-side evidence in ``<episode>/engine``:

* ``replay.jsonl``: the build/platform fingerprint, seed and StartInfo, every
  pack the server's game handler received in order, a running digest of every
  pack the server sent, and logical-state digests at declared boundaries
  (accepted end of turn, battle end, game end).
* ``observations-p<N>.jsonl``: for each player, the client pack position and
  digest of every visible-state capture that produced a visible update, and of
  the capture each policy message ended at.
* ``agents/<color>/bridge.jsonl`` (written by the environment): the raw bridge
  stream each policy received, before engine IDs became opaque handles.

``vcmireplay`` re-runs the recorded inputs through the real game handler and
stops at the first divergence of an outcome, sent-pack digest, state digest or
player capture digest. This module adds the second half of the chain: it folds
each player's received visible updates and checks that their digests equal the
engine-side captures, so the policy's observations are tied to the replayed
state. It also runs tamper tests that must each be detected, and verifies a
series of episodes.

All of this material names engine IDs and hidden state. It is verification
evidence only and never a policy input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_SCORING_SPEC = importlib.util.spec_from_file_location("heroes_episode_scoring", ROOT / "experiments/episodes/scoring.py")
scoring = importlib.util.module_from_spec(_SCORING_SPEC)
_SCORING_SPEC.loader.exec_module(scoring)

SCHEMA = "heroes-replay-v1"
REPORT_SCHEMA = "heroes-replay-verification-v1"
DIGEST_SEED = 1469598103934665603
DIGEST_PRIME = 1099511628211
MASK = (1 << 64) - 1
POLICY_MESSAGES = ("observation", "action_result", "decision", "terminal")
DEFAULT_REPLAY_BINARY = ROOT / "vcmi/build/bin/vcmireplay"


class ReplayError(ValueError):
    pass


# -- player-capture digest -------------------------------------------------
class Digest:
    """FNV-1a over signed 64-bit little-endian values (engine/ExplorerRL/VisibleState.h)."""

    def __init__(self):
        self.state = DIGEST_SEED

    def add(self, value: int) -> None:
        bits = int(value) & MASK
        state = self.state
        for _ in range(8):
            state = ((state ^ (bits & 0xFF)) * DIGEST_PRIME) & MASK
            bits >>= 8
        self.state = state

    def hex(self) -> str:
        return f"{self.state:016x}"


class VisibleFold:
    """One player's visible view, rebuilt only from the visible updates it received."""

    def __init__(self, map_size: list[int]):
        if not isinstance(map_size, list) or len(map_size) != 3 or not all(type(v) is int and v > 0 for v in map_size):
            raise ReplayError(f"Invalid map size {map_size!r}")
        self.width, self.height, self.levels = map_size
        self.tiles: dict[int, tuple[int, int, int]] = {}
        self.objects: dict[int, tuple[int, ...]] = {}
        self.own: dict | None = None
        self.updates = 0

    def _index(self, item) -> int:
        x, y, z = (item["x"], item["y"], item["z"]) if isinstance(item, dict) else item
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.levels):
            raise ReplayError(f"Tile outside the map: {item!r}")
        return x + self.width * (y + self.height * z)

    def apply(self, event: dict) -> None:
        for key in ("appeared", "changed"):
            for item in event[key]:
                self.objects[item["id"]] = tuple(item[k] for k in ("x", "y", "z", "type", "subtype", "owner"))
        for object_id in event["disappeared"]:
            if self.objects.pop(object_id, None) is None:
                raise ReplayError(f"Visible update {event['sequence']} removes unknown object {object_id}")
        for key in ("revealed", "tiles_changed"):
            for item in event[key]:
                self.tiles[self._index(item)] = (item["terrain"], int(bool(item["blocked"])), int(bool(item["visitable"])))
        for position in event["hidden"]:
            if self.tiles.pop(self._index(position), None) is None:
                raise ReplayError(f"Visible update {event['sequence']} hides a tile that was not visible")
        if event.get("own") is not None:
            self.own = event["own"]
        self.updates += 1

    def digest(self) -> str:
        digest = Digest()
        for value in (self.width, self.height, self.levels):
            digest.add(value)
        for index in sorted(self.tiles):
            digest.add(index)
            for value in self.tiles[index]:
                digest.add(value)
        digest.add(len(self.objects))
        for object_id in sorted(self.objects):
            digest.add(object_id)
            for value in self.objects[object_id]:
                digest.add(value)
        own = self.own or {}
        hero = own.get("hero")
        digest.add(hero is not None)
        for value in hero or []:
            digest.add(value)
        digest.add(own.get("movement", 0))
        digest.add(own.get("movement_max", 0))
        resources = own.get("resources", [])
        digest.add(len(resources))
        for value in resources:
            digest.add(value)
        army = own.get("army", [])
        digest.add(len(army))
        for stack in army:
            for key in ("slot", "creature", "count"):
                digest.add(stack[key])
        towns = own.get("towns", [])
        digest.add(len(towns))
        for town in towns:
            for value in town:
                digest.add(value)
        digest.add(own.get("mines", 0))
        return digest.hex()


# -- evidence files ----------------------------------------------------------
def read_jsonl(path: Path, *, allow_partial_tail: bool) -> tuple[list[dict], int]:
    """Complete JSON lines and the number of bytes in an unterminated last line.

    Only an engine stopped by the environment can leave such a tail, so it is
    accepted only when the caller says the recording was cut short.
    """
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    body, tail = data[:cut], data[cut:]
    if tail and not allow_partial_tail:
        raise ReplayError(f"{path.name} ends inside a record")
    records = []
    for number, line in enumerate(body.splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ReplayError(f"{path.name} line {number} is not JSON: {error}") from None
    return records, len(tail)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def complete_replay_records(records: list[dict]) -> tuple[list[dict], int]:
    """The longest prefix that ends with a complete input/applied pair."""
    if len(records) < 2 or records[0].get("kind") != "init" or records[1].get("kind") != "start":
        raise ReplayError("Replay log lacks its init and start records")
    end = 2
    while end + 1 < len(records) and records[end].get("kind") == "input" and records[end + 1].get("kind") == "applied":
        end += 2
    return records[:end], len(records) - end


def cross_check_player(bridge_path: Path, observation_path: Path, *, cut_short: bool) -> dict:
    """Check one player's received visible updates against the engine's capture digests."""
    entries, bridge_tail = read_jsonl(bridge_path, allow_partial_tail=cut_short)
    ledger, ledger_tail = read_jsonl(observation_path, allow_partial_tail=cut_short)
    captures = {record["sequence"]: record for record in ledger if record.get("kind") == "capture"}
    messages = [record for record in ledger if record.get("kind") == "message"]
    if [record["index"] for record in messages] != list(range(len(messages))):
        raise ReplayError(f"{observation_path.name}: message indexes are not consecutive")
    fold = None
    checked_captures = checked_messages = 0
    first_mismatch = None

    def mismatch(where: str, expected, actual) -> None:
        nonlocal first_mismatch
        if first_mismatch is None:
            first_mismatch = {"at": where, "engine": expected, "received": actual}

    for entry in entries:
        if entry.get("direction") != "engine":
            continue
        message = entry["message"]
        kind = message.get("type")
        if kind == "ready":
            fold = VisibleFold(message["map_size"])
            continue
        if kind not in POLICY_MESSAGES or fold is None:
            continue
        for event in message.get("events", []):
            if event.get("kind") != "visible_update":
                continue
            fold.apply(event)
            sequence = event["sequence"]
            capture = captures.get(sequence)
            digest = fold.digest()
            if capture is None:
                mismatch(f"visible update {sequence}", None, digest)
            elif capture["digest"] != digest:
                mismatch(f"visible update {sequence}", capture["digest"], digest)
            checked_captures += 1
        if fold.updates == 0:
            continue  # the engine notes messages only once a capture exists
        if checked_messages >= len(messages):
            mismatch(f"{kind} message {checked_messages}", None, fold.digest())
        else:
            record = messages[checked_messages]
            if record["type"] != kind:
                mismatch(f"message {checked_messages} type", record["type"], kind)
            elif record["digest"] != fold.digest():
                mismatch(f"{kind} message {checked_messages}", record["digest"], fold.digest())
        checked_messages += 1
    engine_only = {"captures": len(captures) - checked_captures, "messages": len(messages) - checked_messages}
    if not cut_short and first_mismatch is None and any(engine_only.values()):
        mismatch("engine-only evidence", engine_only, "a complete episode delivers every capture and message")
    return {"verified": first_mismatch is None, "captures_checked": checked_captures,
            "messages_checked": checked_messages, "engine_only_after_cut": engine_only if cut_short else None,
            "partial_tail_bytes": {"bridge": bridge_tail, "observations": ledger_tail},
            "first_mismatch": first_mismatch}


# -- engine replay -----------------------------------------------------------
def replay_profile(directory: Path, episode_dir: Path, data: Path) -> tuple[dict, Path]:
    """A private VCMI profile with the episode's map copy at the recorded path."""
    profile = directory / "profile"
    if sys.platform == "darwin":
        data_root, config_root = profile, profile / "config"
        environment = {"VCMI_USER_DATA_DIR": str(profile)}
    else:
        data_root, config_root = profile / "data" / "vcmi", profile / "config" / "vcmi"
        environment = {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
                       "XDG_CACHE_HOME": str(profile / "cache")}
    maps = sorted((episode_dir / "profile").rglob("Maps/headless.*"))
    if len(maps) != 1:
        raise ReplayError(f"Expected the episode's one map copy under {episode_dir / 'profile'}")
    if not (data / "Data").is_dir():
        raise ReplayError(f"Missing original game assets: {data / 'Data'}")
    for folder in (data_root / "Maps", config_root):
        folder.mkdir(parents=True, exist_ok=True)
    (data_root / "Data").symlink_to(data / "Data", target_is_directory=True)
    shutil.copyfile(maps[0], data_root / "Maps" / maps[0].name)
    scoring.write_json(config_root / "settings.json", {"general": {"music": 0, "sound": 0},
                                                       "logging": {"loggers": [{"domain": "global", "level": "info"}]}})
    return environment, maps[0]


def run_replay(binary: Path, environment: dict, log: Path, observations: Path | None, report: Path, *,
               prefix: bool, trace: Path | None = None, allow_platform_mismatch: bool = False,
               timeout: float = 3600.0) -> dict:
    command = [str(binary), "--log", str(log), "--report", str(report)]
    if observations is not None:
        command += ["--observations", str(observations)]
    if trace is not None:
        command += ["--trace", str(trace)]
    if prefix:
        command.append("--prefix")
    if allow_platform_mismatch:
        command.append("--allow-platform-mismatch")
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
    started = time.monotonic()
    result = subprocess.run(command, cwd=binary.parent, env={**inherited, **environment}, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    seconds = time.monotonic() - started
    try:
        outcome = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        tail = result.stdout.decode(errors="replace")[-2000:]
        raise ReplayError(f"vcmireplay exited {result.returncode} without a report: {tail}") from None
    if outcome.get("schema") != REPORT_SCHEMA or result.returncode not in (0, 1, 3):
        raise ReplayError(f"Unexpected vcmireplay result {result.returncode}")
    outcome["returncode"], outcome["seconds"] = result.returncode, round(seconds, 3)
    return outcome


def prepare_evidence(episode_dir: Path, directory: Path, cut_short: bool) -> dict:
    """Clean copies of the engine evidence, without a tail an interrupted engine left."""
    engine = episode_dir / "engine"
    records, tail_bytes = read_jsonl(engine / "replay.jsonl", allow_partial_tail=cut_short)
    records, unpaired = complete_replay_records(records)
    if unpaired and not cut_short:
        raise ReplayError(f"A complete episode's replay log has {unpaired} unpaired trailing records")
    observations = directory / "observations"
    observations.mkdir(parents=True)
    observation_tails = {}
    for path in sorted(engine.glob("observations-p*.jsonl")):
        lines, tail = read_jsonl(path, allow_partial_tail=cut_short)
        write_jsonl(observations / path.name, lines)
        observation_tails[path.name] = tail
    write_jsonl(directory / "replay.jsonl", records)
    return {"log": directory / "replay.jsonl", "observations": observations, "records": records,
            "dropped": {"replay_partial_tail_bytes": tail_bytes, "replay_unpaired_records": unpaired,
                        "observation_partial_tail_bytes": observation_tails}}


def expected_terminals(manifest: dict) -> list[dict] | None:
    outcomes = manifest.get("outcomes_by_player") or {}
    if manifest.get("status") != "completed" or set(outcomes.values()) != {"win", "loss"}:
        return None
    return sorted(({"player": index, "victory": outcomes[color] == "win"} for index, color in enumerate(("red", "blue"))),
                  key=lambda entry: entry["player"])


def verify_episode(episode_dir: Path, *, binary: Path, data: Path, output: Path | None = None,
                   trace: bool = False, allow_platform_mismatch: bool = False, timeout: float = 3600.0) -> dict:
    """Replay one episode and cross-check both players' received observations."""
    episode_dir = episode_dir.expanduser().resolve()
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    cut_short = manifest["status"] != "completed"
    directory = (output or episode_dir / "verification").expanduser().resolve()
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    report = {"schema_version": 1, "kind": "headless-replay-verification", "episode": manifest["episode"],
              "seed": manifest["seed"], "status": manifest["status"],
              "termination_reason": manifest["termination_reason"], "created_at": datetime.now(timezone.utc).isoformat(),
              "replay_binary_sha256": scoring.digest(binary), "cut_short": cut_short, "verified": False}
    try:
        evidence = prepare_evidence(episode_dir, directory, cut_short)
        report["dropped"] = evidence["dropped"]
        report["replay_log_sha256"] = scoring.digest(episode_dir / "engine" / "replay.jsonl")
        environment, _ = replay_profile(directory, episode_dir, data)
        engine = run_replay(binary, environment, evidence["log"], evidence["observations"], directory / "engine-report.json",
                            prefix=cut_short, trace=directory / "trace.jsonl" if trace else None,
                            allow_platform_mismatch=allow_platform_mismatch, timeout=timeout)
        report["engine"] = engine
        players = {}
        for index, color in enumerate(("red", "blue")):
            bridge = episode_dir / "agents" / color / "bridge.jsonl"
            observations = episode_dir / "engine" / f"observations-p{index}.jsonl"
            players[color] = (cross_check_player(bridge, observations, cut_short=cut_short)
                              if bridge.is_file() and observations.is_file() else {"verified": False, "missing": True})
        report["players"] = players
        terminals = expected_terminals(manifest)
        summary = engine.get("summary") or {}
        report["terminal_evidence"] = {
            "expected": terminals, "replayed": summary.get("terminals"),
            "matches": terminals is None or sorted(summary.get("terminals") or [], key=lambda e: e["player"]) == terminals,
            "game_end_boundaries": (summary.get("boundaries") or {}).get("game_end", 0)}
        natural_ok = terminals is None or (report["terminal_evidence"]["matches"]
                                           and report["terminal_evidence"]["game_end_boundaries"] >= 1)
        report["verified"] = bool(engine.get("verified") and all(p.get("verified") for p in players.values()) and natural_ok)
    except (ReplayError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    shutil.rmtree(directory / "profile", ignore_errors=True)
    scoring.write_json(directory / "replay-verification.json", report)
    return report


# -- tamper tests --------------------------------------------------------------
def _flip_hex(text: str, position: int) -> str:
    digit = int(text[position], 16) ^ 0x1
    return text[:position] + f"{digit:x}" + text[position + 1:]


def tamper_cases(records: list[dict], observations: dict[str, list[dict]]) -> list[tuple[str, str, list[dict], dict]]:
    """Mutations of a verified recording, each of which the verifier must reject.

    Returns (name, expectation, replay records, observation files) tuples.
    """
    inputs = [index for index in range(2, len(records), 2)]
    if len(inputs) < 4:
        raise ReplayError("Tamper tests need at least four recorded inputs")
    middle = inputs[len(inputs) // 2]
    cases = []

    def copy_records():
        return json.loads(json.dumps(records))

    seed = copy_records()
    seed[0]["seed"] = seed[0]["seed"] + 1
    cases.append(("seed", "the initial state digest no longer matches", seed, observations))

    # An accepted end of turn ends with its request ID, which the server echoes:
    # flipping that byte's lowest bit is a valid pack whose effects must differ.
    turn = next((index for index in inputs[len(inputs) // 2:] if records[index + 1].get("boundary") == "end_turn"), middle)
    flipped = copy_records()
    pack = flipped[turn]["pack"]
    flipped[turn]["pack"] = _flip_hex(pack, len(pack) - 1)
    cases.append(("pack_byte", "a changed input byte changes the decoded pack or its outcome", flipped, observations))

    digested = [index + 1 for index in inputs if records[index + 1].get("digest")]
    if digested:
        state = copy_records()
        target = digested[len(digested) // 2]
        state[target]["digest"] = _flip_hex(state[target]["digest"], 0)
        cases.append(("state_digest", "a recorded boundary digest differs from the replayed state", state, observations))

    dropped = copy_records()
    del dropped[middle:middle + 2]
    for index in range(middle, len(dropped), 2):
        dropped[index]["seq"] = dropped[index + 1]["seq"] = (index - 2) // 2
    cases.append(("drop_input", "a removed input, renumbered, changes later outcomes", dropped, observations))

    swapped = copy_records()
    a, b = next(((i, j) for i, j in zip(inputs, inputs[1:]) if records[i]["pack"] != records[j]["pack"]), (None, None))
    if a is not None:
        for key in ("pack", "type", "player", "connection", "request"):
            swapped[a][key], swapped[b][key] = swapped[b][key], swapped[a][key]
        cases.append(("swap_inputs", "two reordered inputs change an outcome or digest", swapped, observations))

    for name, lines in sorted(observations.items()):
        captures = [i for i, line in enumerate(lines) if line.get("kind") == "capture"]
        if captures:
            changed = json.loads(json.dumps(observations))
            target = captures[len(captures) // 2]
            changed[name][target]["digest"] = _flip_hex(changed[name][target]["digest"], 0)
            cases.append((f"observation_digest_{name.split('.')[0]}", "a recorded player capture differs from the replay",
                          records, changed))
    return cases


def tamper_episode(episode_dir: Path, *, binary: Path, data: Path, output: Path | None = None,
                   allow_platform_mismatch: bool = False, timeout: float = 3600.0) -> dict:
    """Run every tamper case of a verified episode; each must be rejected."""
    episode_dir = episode_dir.expanduser().resolve()
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    cut_short = manifest["status"] != "completed"
    directory = (output or episode_dir / "tamper").expanduser().resolve()
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    evidence = prepare_evidence(episode_dir, directory / "original", cut_short)
    observations = {path.name: read_jsonl(path, allow_partial_tail=False)[0]
                    for path in sorted(evidence["observations"].glob("observations-p*.jsonl"))}
    environment, _ = replay_profile(directory, episode_dir, data)
    results = []
    for name, expectation, records, files in tamper_cases(evidence["records"], observations):
        case = directory / name
        (case / "observations").mkdir(parents=True)
        write_jsonl(case / "replay.jsonl", records)
        for filename, lines in files.items():
            write_jsonl(case / "observations" / filename, lines)
        outcome = run_replay(binary, environment, case / "replay.jsonl", case / "observations", case / "report.json",
                             prefix=cut_short, allow_platform_mismatch=allow_platform_mismatch, timeout=timeout)
        results.append({"case": name, "expectation": expectation, "detected": not outcome.get("verified"),
                        "divergence": outcome.get("divergence"), "error": outcome.get("error"),
                        "returncode": outcome["returncode"]})
        shutil.rmtree(case)
    # The observation chain: an altered received update must break the cross-check.
    bridge_case = _tamper_bridge(episode_dir, directory, cut_short)
    if bridge_case is not None:
        results.append(bridge_case)
    shutil.rmtree(directory / "profile", ignore_errors=True)
    shutil.rmtree(directory / "original", ignore_errors=True)
    report = {"schema_version": 1, "kind": "headless-replay-tamper-tests", "episode": manifest["episode"],
              "created_at": datetime.now(timezone.utc).isoformat(), "cases": results,
              "all_detected": bool(results) and all(result["detected"] for result in results)}
    scoring.write_json(directory / "tamper-report.json", report)
    return report


def _tamper_bridge(episode_dir: Path, directory: Path, cut_short: bool) -> dict | None:
    bridge = episode_dir / "agents" / "red" / "bridge.jsonl"
    observations = episode_dir / "engine" / "observations-p0.jsonl"
    if not bridge.is_file() or not observations.is_file():
        return None
    entries, _ = read_jsonl(bridge, allow_partial_tail=cut_short)
    for entry in entries:
        for event in entry.get("message", {}).get("events", []) if entry.get("direction") == "engine" else []:
            if event.get("kind") == "visible_update" and event.get("own") is not None:
                event["own"]["movement"] += 1
                path = directory / "bridge-red.jsonl"
                write_jsonl(path, entries)
                check = cross_check_player(path, observations, cut_short=cut_short)
                path.unlink()
                return {"case": "received_update_red", "expectation": "an altered received update breaks the capture chain",
                        "detected": not check["verified"], "divergence": check["first_mismatch"], "error": None,
                        "returncode": None}
    return None


# -- command line ----------------------------------------------------------------
def episode_directories(run: Path) -> list[Path]:
    return sorted(path.parent for path in (run / "episodes").glob("episode-*/manifest.json"))


def verify_series(run: Path, *, binary: Path, data: Path, count: int | None, tamper: int, trace: int,
                  allow_platform_mismatch: bool, prefer_natural: bool) -> dict:
    episodes = episode_directories(run)
    if prefer_natural:
        manifests = {path: json.loads((path / "manifest.json").read_text()) for path in episodes}
        episodes = sorted(episodes, key=lambda path: (manifests[path]["status"] != "completed", path.name))
    selected = episodes[:count] if count else episodes
    rows = []
    for position, episode in enumerate(selected):
        result = verify_episode(episode, binary=binary, data=data, trace=position < trace,
                                allow_platform_mismatch=allow_platform_mismatch)
        row = {"episode": result["episode"], "status": result["status"], "termination_reason": result["termination_reason"],
               "verified": result["verified"], "error": result.get("error"),
               "divergence": (result.get("engine") or {}).get("divergence"),
               "summary": (result.get("engine") or {}).get("summary"),
               "players": {color: {k: v for k, v in check.items() if k in ("verified", "captures_checked",
                                                                         "messages_checked", "first_mismatch")}
                           for color, check in (result.get("players") or {}).items()},
               "terminal_evidence": result.get("terminal_evidence"),
               "seconds": (result.get("engine") or {}).get("seconds")}
        if position < tamper and result["verified"]:
            row["tamper"] = tamper_episode(episode, binary=binary, data=data,
                                           allow_platform_mismatch=allow_platform_mismatch)
        rows.append(row)
        print(json.dumps({"episode": row["episode"], "status": row["status"], "verified": row["verified"],
                          "tamper_detected": row.get("tamper", {}).get("all_detected")}), flush=True)
    boundaries = {}
    for row in rows:
        for kind, number in ((row.get("summary") or {}).get("boundaries") or {}).items():
            boundaries[kind] = boundaries.get(kind, 0) + number
    summary = {"schema_version": 1, "kind": "headless-replay-series", "run": str(run),
               "created_at": datetime.now(timezone.utc).isoformat(), "episodes": rows,
               "aggregate": {"verified": sum(row["verified"] for row in rows), "attempted": len(rows),
                             "natural_verified": sum(row["verified"] and row["status"] == "completed" for row in rows),
                             "boundaries": boundaries,
                             "tamper_runs": sum("tamper" in row for row in rows),
                             "tamper_all_detected": all(row["tamper"]["all_detected"] for row in rows if "tamper" in row)}}
    scoring.write_json(run / "replay-series.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "tamper", "series"):
        command = commands.add_parser(name)
        command.add_argument("target", type=Path, help="An episode directory, or a run directory for series")
        command.add_argument("--binary", type=Path, default=DEFAULT_REPLAY_BINARY)
        command.add_argument("--data", type=Path, required=True, help="Directory that contains the game's Data folder")
        command.add_argument("--allow-platform-mismatch", action="store_true")
        if name == "verify":
            command.add_argument("--trace", action="store_true", help="Write the per-input trace for offline rendering")
        if name == "series":
            command.add_argument("--count", type=int, default=None)
            command.add_argument("--tamper", type=int, default=1, help="Run tamper tests on the first N verified episodes")
            command.add_argument("--trace", type=int, default=0, help="Write traces for the first N episodes")
            command.add_argument("--prefer-natural", action="store_true", help="Verify natural endings first")
    args = parser.parse_args(argv)
    binary = args.binary.expanduser().resolve()
    if not binary.is_file() or b"heroes-replay-verification-v1" not in binary.read_bytes():
        parser.exit(1, f"Build the replay verifier first: {binary}\n")
    data = args.data.expanduser().resolve()
    if args.command == "verify":
        result = verify_episode(args.target, binary=binary, data=data, trace=args.trace,
                                allow_platform_mismatch=args.allow_platform_mismatch)
        print(json.dumps({key: result.get(key) for key in ("episode", "status", "verified", "error")}))
        return 0 if result["verified"] else 1
    if args.command == "tamper":
        result = tamper_episode(args.target, binary=binary, data=data,
                                allow_platform_mismatch=args.allow_platform_mismatch)
        print(json.dumps({"all_detected": result["all_detected"],
                          "cases": {case["case"]: case["detected"] for case in result["cases"]}}))
        return 0 if result["all_detected"] else 1
    result = verify_series(args.target.expanduser().resolve(), binary=binary, data=data, count=args.count,
                           tamper=args.tamper, trace=args.trace, allow_platform_mismatch=args.allow_platform_mismatch,
                           prefer_natural=args.prefer_natural)
    print(json.dumps(result["aggregate"], indent=2))
    return 0 if result["aggregate"]["verified"] == result["aggregate"]["attempted"] else 1


if __name__ == "__main__":
    sys.exit(main())
