#!/usr/bin/env python3
"""Offline analysis of recorded Heroes RL attempts. It reads existing run directories only.

It never launches games, changes source runs, or turns a truncation, technical
failure or hero death into a loss. Only a top-level manifest reward that agrees
with every piece of completion evidence counts as a verified natural outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ANALYSIS_SCHEMA_VERSION = 1
SUPPORTED_MANIFESTS = {(2, "random-baseline")}
UNKNOWN = "unknown"
REWARDS = {"win": 1, "loss": -1}
CAVEATS = [
    "Offline analysis of recorded attempts; it does not run games, train or evaluate a policy.",
    "Only verified natural endings are scored. Timeouts, limits, technical failures and inconsistent evidence are unscored.",
    "A reported socket terminal is evidence, not a score; a nested terminal.reward is ignored.",
    "Ledger metrics require a matching hash, player/episode identity and nondecreasing timestamps; a usable ledger terminal must agree with the score.",
    "Losing all heroes is not a game defeat. Hero-loss intervals are bounded by observed ledger states, not exact transition times.",
    "Cohorts with different map, engine, policy, opponent, side or limits are not pooled.",
    "A handful of games does not establish improvement, a win rate or statistical significance.",
]


def diagnostic(level: str, code: str, message: str) -> dict:
    return {"level": level, "code": code, "message": message}


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def value_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_int(value) -> bool:
    return type(value) is int


def is_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def field(value, *keys, default=UNKNOWN):
    """Nested lookup that keeps an absent field visibly unknown, distinct from an explicit null."""
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


# --- Cohorts and seeds ---------------------------------------------------------------------------

def cohort_components(manifest: dict) -> dict:
    """Everything that must match before attempts are comparable. Absent metadata stays unknown."""
    engine = field(manifest, "engine")
    recipe = field(manifest, "engine_recipe")
    libraries = field(engine, "ai_libraries")
    limits = field(manifest, "limits")
    return {
        "map": {"name": field(manifest, "map", "name"), "sha256": field(manifest, "map", "sha256")},
        "engine": {"commit": field(engine, "commit"), "dirty": field(engine, "dirty"),
                   "binary_sha256": field(engine, "binary_sha256"),
                   "ai_libraries_digest": value_digest(libraries) if isinstance(libraries, dict) else UNKNOWN},
        "engine_recipe_digest": value_digest(recipe) if isinstance(recipe, dict) else UNKNOWN,
        "policy": {"algorithm": field(manifest, "policy", "algorithm"), "label": field(manifest, "policy", "label"),
                   "checkpoint": field(manifest, "policy", "checkpoint"),
                   "learning": field(manifest, "policy", "learning"),
                   "action_space": field(manifest, "policy", "action_space"),
                   "dialog_policy": field(manifest, "policy", "dialog_policy"),
                   "step_delay_seconds": field(manifest, "policy", "step_delay_seconds"),
                   "source_sha256": field(manifest, "policy_source", "sha256")},
        "opponent": {"module": field(manifest, "opponent", "module"),
                     "difficulty": field(manifest, "opponent", "difficulty"),
                     "difficulty_index": field(manifest, "opponent", "difficulty_index"),
                     "combat": field(manifest, "combat")},
        "selected_side": {"agent_player_index": field(manifest, "agent_player_index"),
                          "outcome_player": field(manifest, "outcome_player")},
        "limits": limits if isinstance(limits, dict) else UNKNOWN,
        "capture_fps_by_view": {
            name: field(view, "capture", "fps")
            for name, view in sorted(as_dict(field(manifest, "recordings")).items())},
    }


def unknown_paths(value, prefix: str = "") -> list[str]:
    if value == UNKNOWN:
        return [prefix]
    if isinstance(value, dict):
        return [path for key in sorted(value) for path in unknown_paths(value[key], f"{prefix}.{key}" if prefix else key)]
    return []


def seeds(manifest: dict, diagnostics: list) -> dict:
    engine_seed = field(manifest, "seed")
    policy_seed = field(manifest, "policy", "seed")
    effective = field(manifest, "effective_engine_rng", "initial_seed")
    for name, value in (("engine", engine_seed), ("policy", policy_seed)):
        if value != UNKNOWN and not is_int(value):
            diagnostics.append(diagnostic("warning", f"{name}_seed_invalid", f"{name} seed is not an integer: {value!r}"))
    if is_int(engine_seed) and is_int(effective) and effective != engine_seed:
        diagnostics.append(diagnostic("warning", "engine_seed_mismatch",
                                      f"Requested engine seed {engine_seed} differs from logged initial seed {effective}"))
    return {"engine_seed": engine_seed if is_int(engine_seed) else UNKNOWN,
            "policy_seed": policy_seed if is_int(policy_seed) else UNKNOWN,
            "effective_engine_seed": effective if is_int(effective) else UNKNOWN}


# --- Outcome verification ------------------------------------------------------------------------

def verification_checks(manifest: dict, ledger: dict | None = None) -> list[dict]:
    """Every condition required before the manifest reward may be counted."""
    execution = field(manifest, "policy_execution")
    status, reason = field(manifest, "status"), field(manifest, "termination_reason")
    outcome, reward = field(manifest, "outcome"), field(manifest, "reward")
    recordings = field(manifest, "recordings")
    terminal = field(execution, "terminal")
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("natural_completion", status == "completed" and reason == "natural_terminal",
          f"status={status!r}, termination_reason={reason!r}")
    returncode = field(manifest, "returncode")
    check("clean_exit", is_int(returncode) and returncode == 0, f"returncode={returncode!r}")
    views = recordings if isinstance(recordings, dict) else {}
    check("valid_recording", field(manifest, "recording_valid") is True and bool(views)
          and all(field(view, "recording_valid") is True for view in views.values()),
          f"recording_valid={field(manifest, 'recording_valid')!r}, views="
          + (", ".join(f"{name}:{field(view, 'recording_valid')!r}" for name, view in sorted(views.items())) or "none"))
    actions = field(execution, "actions_sent")
    check("agent_ready_and_acted", field(execution, "ready") is True and is_int(actions) and actions >= 1,
          f"ready={field(execution, 'ready')!r}, actions_sent={actions!r}")
    error, limit = field(execution, "error"), field(execution, "limit")
    check("no_policy_error_or_limit", error is None and limit is None, f"error={error!r}, limit={limit!r}")
    valid_outcome = isinstance(outcome, str) and outcome in REWARDS
    check("natural_outcome_value", valid_outcome, f"outcome={outcome!r}")
    reported = field(terminal, "outcome")
    check("socket_terminal_matches", valid_outcome and reported == outcome,
          f"manifest outcome={outcome!r}, socket terminal outcome={reported!r}")
    check("reward_matches_outcome", valid_outcome and is_int(reward) and reward == REWARDS[outcome],
          f"top-level reward={reward!r} for outcome={outcome!r}")
    # A ledger is supplementary, but a usable ledger must not contradict the score.
    if ledger and ledger.get("used_for_metrics") and ledger.get("terminal_outcome") is not None:
        check("ledger_terminal_matches", valid_outcome and ledger["terminal_outcome"] == outcome,
              f"manifest outcome={outcome!r}, ledger terminal outcome={ledger['terminal_outcome']!r}")
    return checks


def classify(manifest: dict, checks: list[dict], diagnostics: list) -> str:
    status, outcome, reward = field(manifest, "status"), field(manifest, "outcome"), field(manifest, "reward")
    if all(item["passed"] for item in checks):
        return "verified_natural"
    if status == "completed":
        failed = ", ".join(item["check"] for item in checks if not item["passed"])
        diagnostics.append(diagnostic("error", "unverified_completion",
                                      f"Manifest claims completion but evidence disagrees: {failed}"))
        return "inconsistent"
    if outcome not in (None, UNKNOWN) or reward not in (None, UNKNOWN):
        diagnostics.append(diagnostic("error", "unscored_attempt_has_result",
                                      f"status={status!r} must not carry outcome={outcome!r} or reward={reward!r}"))
        return "inconsistent"
    if status == "timeout":
        return "truncated"
    if status == "failed":
        return "technical_failure"
    diagnostics.append(diagnostic("error", "unrecognized_status", f"Unrecognized manifest status {status!r}"))
    return "unrecognized_status"


# --- Policy ledger -------------------------------------------------------------------------------

def observed_point(snapshot: dict) -> dict:
    return {key: snapshot[key] for key in ("elapsed_seconds", "source", "day", "own_heroes", "own_towns")}


def observed_intervals(snapshots: list[dict], condition: str, predicate) -> list[dict]:
    """Runs of consecutive observed states satisfying a predicate, bounded by neighbouring observations."""
    intervals, index = [], 0
    while index < len(snapshots):
        if not predicate(snapshots[index]):
            index += 1
            continue
        end = index
        while end + 1 < len(snapshots) and predicate(snapshots[end + 1]):
            end += 1
        intervals.append({
            "condition": condition,
            "first_observed": observed_point(snapshots[index]),
            "last_observed": observed_point(snapshots[end]),
            "preceding_observation": observed_point(snapshots[index - 1]) if index else None,
            "following_observation": observed_point(snapshots[end + 1]) if end + 1 < len(snapshots) else None,
            "left_censored": index == 0, "right_censored": end + 1 == len(snapshots)})
        index = end + 1
    return intervals


def analyze_ledger(run: Path, manifest: dict, diagnostics: list) -> dict:
    declared = field(manifest, "policy_ledger")
    name = field(declared, "file", default="policy.jsonl")
    result = {"declared": isinstance(declared, dict), "file": name, "present": False,
              "declared_sha256": field(declared, "sha256"), "actual_sha256": None,
              "hash_verified": None, "parsed": False, "used_for_metrics": False}
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        diagnostics.append(diagnostic("error", "ledger_path_invalid", f"Ledger file must be a plain name: {name!r}"))
        return result
    path = run / name
    if not path.is_file():
        if result["declared"]:
            diagnostics.append(diagnostic("warning", "ledger_missing", f"Declared policy ledger is absent: {name}"))
        return result
    result["present"] = True
    result["actual_sha256"] = file_digest(path)
    if not result["declared"]:
        diagnostics.append(diagnostic("warning", "ledger_undeclared", "Policy ledger exists but the manifest declares no hash"))
        return result
    result["hash_verified"] = result["actual_sha256"] == result["declared_sha256"]
    if not result["hash_verified"]:
        diagnostics.append(diagnostic("error", "ledger_hash_mismatch",
                                      "Policy ledger hash differs from the manifest; ledger metrics are withheld"))
        return result
    entries = []
    previous_elapsed = 0.0
    expected_player, expected_episode = field(manifest, "agent_player_index"), field(manifest, "episode")
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            diagnostics.append(diagnostic("error", "ledger_malformed", f"Ledger line {number} is not JSON: {error}"))
            return result
        if (not isinstance(entry, dict) or not is_number(entry.get("elapsed_seconds"))
                or not isinstance(entry.get("direction"), str)
                or entry["direction"] not in {"engine", "policy"} or not isinstance(entry.get("message"), dict)):
            diagnostics.append(diagnostic("error", "ledger_malformed",
                                          f"Ledger line {number} is not {{elapsed_seconds, direction, message}}"))
            return result
        elapsed, message = entry["elapsed_seconds"], entry["message"]
        if elapsed < previous_elapsed:
            diagnostics.append(diagnostic("error", "ledger_time_invalid",
                                          f"Ledger line {number} has a negative or decreasing timestamp"))
            return result
        previous_elapsed = elapsed
        if entry["direction"] == "engine":
            if not is_int(expected_player) or not (is_int(expected_episode) or isinstance(expected_episode, str)):
                diagnostics.append(diagnostic("error", "ledger_identity_unknown",
                                              "Manifest must identify the player and episode to use ledger metrics"))
                return result
            if (not is_int(message.get("player")) or message["player"] != expected_player
                    or message.get("episode") != str(expected_episode)):
                diagnostics.append(diagnostic("error", "ledger_identity_mismatch",
                                              f"Ledger line {number} has the wrong player or episode"))
                return result
            kind = message.get("type")
            if not isinstance(kind, str):
                diagnostics.append(diagnostic("error", "ledger_malformed",
                                              f"Ledger line {number} has no string message type"))
                return result
            if kind in {"action_result", "terminal"} and "events" in message:
                event_list = message["events"]
                if (not isinstance(event_list, list)
                        or any(not isinstance(event, dict) or not isinstance(event.get("kind"), str)
                               for event in event_list)):
                    diagnostics.append(diagnostic("error", "ledger_malformed",
                                                  f"Ledger line {number} has invalid event kinds"))
                    return result
            if kind == "terminal" and (not isinstance(message.get("outcome"), str)
                                       or message["outcome"] not in REWARDS):
                diagnostics.append(diagnostic("error", "ledger_malformed",
                                              f"Ledger line {number} has an invalid terminal outcome"))
                return result
        entries.append(entry)
    result["parsed"] = True
    result["entries"] = len(entries)
    snapshots, events, terminal_outcome = [], Counter(), None

    def add(state, elapsed, source):
        if isinstance(state, dict) and is_int(state.get("own_heroes")) and is_int(state.get("own_towns")):
            snapshots.append({"elapsed_seconds": elapsed, "source": source, "day": state.get("day"),
                              "own_heroes": state["own_heroes"], "own_towns": state["own_towns"],
                              "own_mines": state.get("own_mines")})

    for entry in entries:
        message, elapsed = entry["message"], entry["elapsed_seconds"]
        if entry["direction"] != "engine":
            continue
        kind = message.get("type")
        if kind == "observation":
            add(message.get("state"), elapsed, "observation.state")
        elif kind == "action_result":
            add(message.get("before"), elapsed, "action_result.before")
            add(message.get("after"), elapsed, "action_result.after")
        elif kind == "terminal":
            add(message.get("state"), elapsed, "terminal.state")
            terminal_outcome = message.get("outcome")
        if kind in {"action_result", "terminal"} and isinstance(message.get("events"), list):
            events.update(event.get("kind", UNKNOWN) for event in message["events"] if isinstance(event, dict))
    reported = field(manifest, "policy_execution", "terminal", "outcome")
    if terminal_outcome is not None and reported != terminal_outcome:
        diagnostics.append(diagnostic("error", "ledger_terminal_mismatch",
                                      f"Ledger terminal {terminal_outcome!r} differs from execution terminal {reported!r}"))
    mines = [s["own_mines"] for s in snapshots if is_int(s["own_mines"])]
    result.update(used_for_metrics=True, state_observations=len(snapshots), terminal_outcome=terminal_outcome,
                  event_counts=dict(sorted(events.items())), max_own_mines_observed=max(mines) if mines else None,
                  zero_hero_intervals=observed_intervals(snapshots, "no_own_heroes", lambda s: s["own_heroes"] == 0),
                  town_only_intervals=observed_intervals(
                      snapshots, "no_own_heroes_with_towns", lambda s: s["own_heroes"] == 0 and s["own_towns"] > 0))
    return result


# --- Metrics -------------------------------------------------------------------------------------

def metrics(manifest: dict, ledger: dict) -> dict:
    """Recorded quantities only; absent ones stay null and are listed as missing."""
    execution = field(manifest, "policy_execution")
    initial, last = field(execution, "initial_observation"), field(execution, "last_observation")
    terminal_state = field(execution, "terminal", "state")

    def number(value):
        return value if is_int(value) else None

    def state_value(key):
        for source, state in (("terminal.state", terminal_state), ("last_observation", last)):
            if is_int(field(state, key)):
                return state[key], source
        return None, None

    final_tiles, tiles_source = state_value("visible_tiles")
    first_tiles = number(field(initial, "visible_tiles"))
    battles, battles_source = state_value("completed_battles")
    mines, mines_source = state_value("own_mines")
    result = {key: number(field(execution, key)) for key in (
        "actions_chosen", "actions_sent", "actions_accepted", "actions_rejected", "decisions_sent",
        "successful_displacements", "unique_hero_positions", "completed_turns")}
    result.update(
        game_days={"initial_observed": number(field(initial, "day")), "last_observed": number(field(execution, "last_day")),
                   "terminal": number(field(terminal_state, "day"))},
        visible_tiles={"initial": first_tiles, "final": final_tiles, "final_source": tiles_source,
                       "expansion": final_tiles - first_tiles if first_tiles is not None and final_tiles is not None else None},
        completed_battles={"value": battles, "source": battles_source},
        own_mines={"initial": number(field(initial, "own_mines")), "final": mines, "final_source": mines_source,
                   "max_observed_in_ledger": ledger.get("max_own_mines_observed") if ledger.get("used_for_metrics") else None},
        ledger_event_counts=ledger.get("event_counts") if ledger.get("used_for_metrics") else None)
    missing = [key for key, value in result.items() if value is None]
    for group in ("game_days", "visible_tiles", "own_mines"):
        missing += [f"{group}.{key}" for key, value in result[group].items() if value is None and not key.endswith("_source")]
    if result["completed_battles"]["value"] is None:
        missing.append("completed_battles")
    result["missing"] = sorted(missing)
    return result


# --- Attempts and summary ------------------------------------------------------------------------

def load_attempt(raw: str, number: int, seen: dict) -> tuple[dict, dict | None, Path]:
    run = Path(raw).expanduser().resolve()
    attempt = {"attempt": number, "input": raw, "path": str(run), "validity": None, "diagnostics": []}
    diagnostics = attempt["diagnostics"]
    if run in seen:
        attempt["validity"] = "duplicate"
        diagnostics.append(diagnostic("error", "duplicate_input", f"Same directory as attempt {seen[run]}; counted once"))
        return attempt, None, run
    seen[run] = number
    if not run.is_dir():
        attempt["validity"] = "missing"
        diagnostics.append(diagnostic("error", "run_missing", "Run directory does not exist"))
        return attempt, None, run
    path = run / "manifest.json"
    if not path.is_file():
        attempt["validity"] = "missing"
        diagnostics.append(diagnostic("error", "manifest_missing", "Run directory has no manifest.json"))
        return attempt, None, run
    attempt["manifest_sha256"] = file_digest(path)
    try:
        manifest = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        attempt["validity"] = "malformed"
        diagnostics.append(diagnostic("error", "manifest_not_json", f"manifest.json is not valid JSON: {error}"))
        return attempt, None, run
    if not isinstance(manifest, dict):
        attempt["validity"] = "malformed"
        diagnostics.append(diagnostic("error", "manifest_not_object", "manifest.json is not a JSON object"))
        return attempt, None, run
    schema, kind = manifest.get("schema_version"), manifest.get("kind")
    attempt.update(schema_version=schema, kind=kind)
    if not is_int(schema) or not isinstance(kind, str):
        attempt["validity"] = "malformed"
        diagnostics.append(diagnostic("error", "manifest_schema_invalid",
                                      "Manifest schema_version must be an integer and kind must be a string"))
        return attempt, None, run
    if (schema, kind) not in SUPPORTED_MANIFESTS:
        attempt["validity"] = "unsupported"
        diagnostics.append(diagnostic("error", "manifest_unsupported",
                                      f"Unsupported manifest schema_version={schema!r}, kind={kind!r}; "
                                      "supported: schema-v2 random-baseline"))
        return attempt, None, run
    if not isinstance(manifest.get("status"), str):
        attempt["validity"] = "malformed"
        diagnostics.append(diagnostic("error", "status_missing", "Manifest has no string status"))
        return attempt, None, run
    for key in ("policy_execution", "recordings"):
        if key in manifest and not isinstance(manifest[key], dict):
            diagnostics.append(diagnostic("error", f"{key}_invalid", f"{key} is not an object; treated as absent"))
    attempt["validity"] = "eligible"
    return attempt, manifest, run


def analyze_attempt(raw: str, number: int, seen: dict) -> dict:
    attempt, manifest, run = load_attempt(raw, number, seen)
    if manifest is None:
        return attempt
    diagnostics = attempt["diagnostics"]
    if manifest.get("synthetic_fixture") is True:
        attempt["synthetic_fixture"] = True
        diagnostics.append(diagnostic("info", "synthetic_fixture", "Synthetic test fixture, not game evidence"))
    terminal = field(manifest, "policy_execution", "terminal")
    if isinstance(terminal, dict) and "reward" in terminal:
        diagnostics.append(diagnostic("info", "legacy_terminal_reward_ignored",
                                      "Nested policy_execution.terminal.reward is ignored; only top-level reward counts"))
    ledger = analyze_ledger(run, manifest, diagnostics)
    checks = verification_checks(manifest, ledger)
    classification = classify(manifest, checks, diagnostics)
    components = cohort_components(manifest)
    reported = field(terminal, "outcome", default=None)
    attempt.update(
        episode=field(manifest, "episode"),
        status=manifest["status"], termination_reason=field(manifest, "termination_reason"),
        manifest_outcome=field(manifest, "outcome"), manifest_reward=field(manifest, "reward"),
        classification=classification, verification=checks,
        scored_outcome=manifest["outcome"] if classification == "verified_natural" else None,
        scored_reward=manifest["reward"] if classification == "verified_natural" else None,
        reported_terminal_outcome=reported,
        reported_terminal_unscored=reported is not None and classification != "verified_natural",
        seeds=seeds(manifest, diagnostics),
        cohort_id="cohort-" + value_digest(components)[:12], cohort_components=components,
        unknown_metadata=unknown_paths(components), ledger=ledger, metrics=metrics(manifest, ledger))
    if attempt["unknown_metadata"]:
        diagnostics.append(diagnostic("warning", "metadata_unknown",
                                      "Missing cohort metadata: " + ", ".join(attempt["unknown_metadata"])))
    return attempt


def tally(attempts: list[dict], denominator_label: str) -> dict:
    eligible = [a for a in attempts if a["validity"] == "eligible"]
    verified = [a for a in eligible if a["classification"] == "verified_natural"]
    by = lambda kind: dict(sorted(Counter(str(a["termination_reason"]) for a in eligible
                                          if a["classification"] == kind).items()))
    return {"eligible": len(eligible),
            "classification_counts": dict(sorted(Counter(a["classification"] for a in eligible).items())),
            "natural_completion": {"verified_natural": len(verified), "denominator": len(eligible),
                                   "denominator_definition": denominator_label,
                                   "frequency": len(verified) / len(eligible) if eligible else None},
            "verified_outcomes": {"win": sum(a["scored_outcome"] == "win" for a in verified),
                                  "loss": sum(a["scored_outcome"] == "loss" for a in verified)},
            "truncations_by_reason": by("truncated"), "technical_failures_by_reason": by("technical_failure"),
            "inconsistent_by_reason": by("inconsistent"), "unrecognized_by_reason": by("unrecognized_status")}


def summarize(attempts: list[dict]) -> dict:
    eligible = [a for a in attempts if a["validity"] == "eligible"]
    cohorts = {}
    for attempt in eligible:
        cohorts.setdefault(attempt["cohort_id"], []).append(attempt)
    cohort_reports = [{"cohort_id": cohort, "components": members[0]["cohort_components"],
                       "unknown_metadata": members[0]["unknown_metadata"],
                       "attempts": [a["attempt"] for a in members],
                       **tally(members, "eligible attempts in this cohort")}
                      for cohort, members in cohorts.items()]
    pairs = {}
    for attempt in eligible:
        pair = (attempt["seeds"]["engine_seed"], attempt["seeds"]["policy_seed"])
        if UNKNOWN not in pair:
            pairs.setdefault(pair, []).append(attempt)
    repeated = [{"engine_seed": pair[0], "policy_seed": pair[1], "attempts": [a["attempt"] for a in members],
                 "cohorts": sorted({a["cohort_id"] for a in members}),
                 "classifications": [a["classification"] for a in members]}
                for pair, members in pairs.items() if len(members) > 1]
    overall = tally(attempts, "eligible attempts (supported manifests) across all cohorts")
    if len(cohort_reports) > 1:
        overall["verified_outcomes"] = None
        overall["verified_outcomes_note"] = (f"Not pooled: {len(cohort_reports)} incompatible cohorts; "
                                             "see per-cohort outcomes")
    return {"supplied": len(attempts), "eligible": len(eligible), "invalid": len(attempts) - len(eligible),
            "invalid_by_reason": dict(sorted(Counter(a["validity"] for a in attempts
                                                     if a["validity"] != "eligible").items())),
            "overall": overall, "cohorts": cohort_reports, "repeated_seed_pairs": repeated,
            "unseeded_attempts": [a["attempt"] for a in eligible if UNKNOWN in (a["seeds"]["engine_seed"],
                                                                                a["seeds"]["policy_seed"])],
            "synthetic_fixture_attempts": [a["attempt"] for a in attempts if a.get("synthetic_fixture")]}


def analyze(runs: list[str]) -> dict:
    seen = {}
    attempts = [analyze_attempt(raw, number, seen) for number, raw in enumerate(runs, 1)]
    source = Path(__file__).resolve()
    return {"schema_version": ANALYSIS_SCHEMA_VERSION, "kind": "episode-analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analyzer": {"file": "experiments/analysis/analyze_runs.py", "sha256": file_digest(source)},
            "supported_manifests": [{"schema_version": s, "kind": k} for s, k in sorted(SUPPORTED_MANIFESTS)],
            "caveats": CAVEATS, "summary": summarize(attempts), "attempts": attempts}


# --- Markdown ------------------------------------------------------------------------------------

def cell(value) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def frequency(block: dict) -> str:
    natural = block["natural_completion"]
    rate = "n/a" if natural["frequency"] is None else f"{natural['frequency']:.0%}"
    return f"{natural['verified_natural']} / {natural['denominator']} {natural['denominator_definition']} ({rate})"


def reasons(counts: dict) -> str:
    return ", ".join(f"{reason}: {count}" for reason, count in counts.items()) or "none"


def markdown(report: dict) -> str:
    summary, overall = report["summary"], report["summary"]["overall"]
    lines = ["# Heroes RL offline episode analysis", "",
             f"Generated {report['generated_at']} by `{report['analyzer']['file']}` "
             f"(sha256 `{report['analyzer']['sha256'][:12]}`).", ""]
    if summary["synthetic_fixture_attempts"]:
        lines += [f"**Synthetic fixtures present (attempts {cell(summary['synthetic_fixture_attempts'])}). "
                  "They are test inputs, not game evidence.**", ""]
    lines += ["## Caveats", ""] + [f"- {caveat}" for caveat in report["caveats"]] + [""]
    lines += ["## Attempt counts", "",
              f"- Supplied: {summary['supplied']}; eligible: {summary['eligible']}; invalid: {summary['invalid']}"
              f" ({reasons(summary['invalid_by_reason'])})",
              f"- Verified natural completion: {frequency(overall)}",
              f"- Truncations: {reasons(overall['truncations_by_reason'])}",
              f"- Technical failures: {reasons(overall['technical_failures_by_reason'])}",
              f"- Inconsistent evidence: {reasons(overall['inconsistent_by_reason'])}"]
    if overall["verified_outcomes"] is None:
        lines.append(f"- Verified outcomes: {overall['verified_outcomes_note']}")
    else:
        lines.append(f"- Verified outcomes: {overall['verified_outcomes']['win']} win, "
                     f"{overall['verified_outcomes']['loss']} loss")
    lines += ["", "## Cohorts", ""]
    if not summary["cohorts"]:
        lines += ["No eligible attempts.", ""]
    for cohort in summary["cohorts"]:
        c = cohort["components"]
        lines += [f"### {cohort['cohort_id']}", "",
                  f"- Attempts: {cell(cohort['attempts'])}",
                  f"- Map: {cell(c['map']['name'])} (`{cell(c['map']['sha256'])}`)",
                  f"- Engine: commit {cell(c['engine']['commit'])}, dirty {cell(c['engine']['dirty'])}, "
                  f"binary `{cell(c['engine']['binary_sha256'])}`; recipe digest `{cell(c['engine_recipe_digest'])}`",
                  f"- Policy: {cell(c['policy']['label'])}; algorithm {cell(c['policy']['algorithm'])}; "
                  f"learning {cell(c['policy']['learning'])}; checkpoint {cell(c['policy']['checkpoint'])}; "
                  f"source `{cell(c['policy']['source_sha256'])}`",
                  f"- Pacing: policy delay {cell(c['policy']['step_delay_seconds'])} seconds; "
                  f"capture FPS by view {cell(json.dumps(c['capture_fps_by_view'], sort_keys=True))}",
                  f"- Opponent: {cell(c['opponent']['module'])} {cell(c['opponent']['difficulty'])} "
                  f"(index {cell(c['opponent']['difficulty_index'])})",
                  f"- Agent side: player index {cell(c['selected_side']['agent_player_index'])} "
                  f"({cell(c['selected_side']['outcome_player'])})",
                  f"- Limits: {cell(json.dumps(c['limits'], sort_keys=True) if isinstance(c['limits'], dict) else c['limits'])}",
                  f"- Unknown metadata: {', '.join(cohort['unknown_metadata']) or 'none'}",
                  f"- Verified natural completion: {frequency(cohort)}",
                  f"- Verified outcomes: {cohort['verified_outcomes']['win']} win, {cohort['verified_outcomes']['loss']} loss",
                  f"- Truncations: {reasons(cohort['truncations_by_reason'])}; technical failures: "
                  f"{reasons(cohort['technical_failures_by_reason'])}; inconsistent: {reasons(cohort['inconsistent_by_reason'])}",
                  ""]
    lines += ["## Seeds", ""]
    if summary["repeated_seed_pairs"]:
        lines += [f"- Repeated pair engine {p['engine_seed']} / policy {p['policy_seed']}: attempts "
                  f"{cell(p['attempts'])} ({', '.join(p['classifications'])}); cohorts {', '.join(p['cohorts'])}"
                  for p in summary["repeated_seed_pairs"]]
    else:
        lines.append("- No repeated engine/policy seed pairs among eligible attempts.")
    if summary["unseeded_attempts"]:
        lines.append(f"- Attempts with an unknown seed: {cell(summary['unseeded_attempts'])}")
    lines += ["", "## Attempts", "",
              "| # | Input | Cohort | Engine seed | Policy seed | Status | Termination | Classification | "
              "Scored | Reported terminal | Actions | Rejected | Moves | Positions | Days | Fog | Battles | Mines |",
              "| " + " | ".join(["---"] * 18) + " |"]
    for a in report["attempts"]:
        if a["validity"] != "eligible":
            lines.append(f"| {a['attempt']} | {cell(a['input'])} | — | — | — | — | — | invalid: {a['validity']} | "
                         "— | — | — | — | — | — | — | — | — | — |")
            continue
        m = a["metrics"]
        days, tiles = m["game_days"], m["visible_tiles"]
        day_end = days["terminal"] if days["terminal"] is not None else days["last_observed"]
        reported = (f"{a['reported_terminal_outcome']} (unscored)" if a["reported_terminal_unscored"]
                    else a["reported_terminal_outcome"])
        lines.append("| " + " | ".join(cell(v) for v in (
            a["attempt"], a["input"], a["cohort_id"], a["seeds"]["engine_seed"], a["seeds"]["policy_seed"],
            a["status"], a["termination_reason"], a["classification"], a["scored_outcome"], reported,
            m["actions_sent"], m["actions_rejected"], m["successful_displacements"], m["unique_hero_positions"],
            f"{cell(days['initial_observed'])} → {cell(day_end)}",
            f"{cell(tiles['initial'])} → {cell(tiles['final'])}", m["completed_battles"]["value"],
            m["own_mines"]["final"])) + " |")
    lines += ["", "Fog is the selected side's visible tile count. Missing values are shown as —, never estimated.", "",
              "## Hero loss and town-only survival", "",
              "Observed intervals come from ledger states only (`observation.state`, `action_result.before/after`, "
              "`terminal.state`). The transition happened between the preceding and first observed state; "
              "a censored interval has no observation on that side. Hero loss is not a game defeat.", ""]
    found = False
    for a in report["attempts"]:
        ledger = a.get("ledger") or {}
        for interval in ledger.get("zero_hero_intervals") or []:
            found = True
            first, last, after = interval["first_observed"], interval["last_observed"], interval["following_observation"]
            towns = sorted({first["own_towns"], last["own_towns"]})
            lines.append(f"- Attempt {a['attempt']}: no own hero from day {cell(first['day'])} "
                         f"(t={first['elapsed_seconds']:.3f}s, {first['source']}) to day {cell(last['day'])} "
                         f"(t={last['elapsed_seconds']:.3f}s, {last['source']}); own towns observed {towns}; "
                         + (f"next observation t={after['elapsed_seconds']:.3f}s with {after['own_heroes']} hero(es), "
                            f"{after['own_towns']} town(s)" if after else "right-censored at ledger end")
                         + ("; left-censored" if interval["left_censored"] else ""))
    if not found:
        lines.append("- No zero-hero interval observed in verified ledgers.")
    lines += ["", "## Diagnostics", ""]
    for a in report["attempts"]:
        for d in a["diagnostics"]:
            lines.append(f"- Attempt {a['attempt']} [{d['level']}] `{d['code']}`: {d['message']}")
    if not any(a["diagnostics"] for a in report["attempts"]):
        lines.append("- None.")
    return "\n".join(lines) + "\n"


# --- CLI -----------------------------------------------------------------------------------------

def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        target.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run directories to analyse, in reporting order")
    parser.add_argument("--json", type=Path, help="New machine-readable summary file")
    parser.add_argument("--markdown", type=Path, help="New English Markdown report file")
    args = parser.parse_args(argv)
    outputs = [path.expanduser().resolve() for path in (args.json, args.markdown) if path is not None]
    if len(set(outputs)) != len(outputs):
        parser.error("--json and --markdown must be different files")
    for output in outputs:
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")
        for raw in args.runs:
            run = Path(raw).expanduser().resolve()
            if output == run or run in output.parents:
                parser.error(f"Output must not be written inside a source run: {output}")
    report = analyze(args.runs)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        write_new(args.json.expanduser(), text)
    if args.markdown:
        write_new(args.markdown.expanduser(), markdown(report))
    if not outputs:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
