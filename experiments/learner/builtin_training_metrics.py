"""Offline descriptive evaluation metrics; never a learning or action input.

``per_episode`` checks recorded own-history bindings, not native replay. The
caller supplies the separately verified attempt status to ``aggregate``. A
failed attempt stays failed even when its capture contains a natural terminal.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path

from experiments.headless import builtin_evidence as evidence
from experiments.learner import builtin_behavior as behavior
from experiments.learner import evaluation_behavior as shared

SCHEMA = "heroes-builtin-training-metrics-v1"
AGGREGATE_SCHEMA = "heroes-builtin-training-metrics-aggregate-v1"
LIMITATIONS = [
    "Metrics are descriptive only: no reward, policy input, action selection, optimizer samples or best-checkpoint selection.",
    "Natural win rate conditions on verified natural endings; cutoffs, failures and unstarted attempts are separate, never losses or draws.",
    "Wilson intervals are descriptive binomial intervals. Reused evaluation seeds and paired seats can be dependent; the intervals do not account for that dependence.",
    "Repeated evaluation, monitoring and stopping on results create selection/stopping bias; repeated fixed evaluation seeds are not fresh held-out evidence.",
    "Creature counts are units, not combat strength. Largest displayed-army share is a count concentration measure, not a strategic quality label.",
    "Only currently displayed own army rows are counted. First/last observed rows need not describe episode start/end; closed panels are not refreshed from memory.",
    "Town roster removal establishes loss of ownership, not its cause. Captures require the existing visible visit/owner-change corroboration.",
    "Accepted recruitment quantity follows the exact-count native action/result contract; a separately observed army delta is reported only when both rows are displayed.",
    "Offline history checks do not replace native replay, policy reconstruction or the runner's durable verification and cleanup receipt.",
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshot(frame, row):
    counts = shared._counts(row)
    return {"seq": frame["seq"], "day": frame["day"], "subject": row["subject"], "role": row["role"],
            "creature_count": sum(counts.values()),
            "by_creature": [{"creature": creature, "count": counts[creature]} for creature in sorted(counts)]}


def _own_rows(frame):
    obs = frame["observation"]
    owned = {entity["handle"] for name in ("heroes", "towns") for entity in obs[name]}
    rows = shared._rows(obs)
    _require(set(rows) <= owned, "Displayed army is not bound to a current own entity")
    return rows


def _extras(frames, basic):
    """Input has already passed the sole-player history parser in per_episode."""
    observations = [frame for frame in frames if frame.get("observation") is not None]
    observation_seqs = [frame["seq"] for frame in observations]
    decisions = {frame["seq"]: frame for frame in frames if frame["kind"] == "decision"}
    results = {frame["decision_seq"]: frame for frame in frames if frame["kind"] == "result"}
    submissions = [frame for frame in frames if frame["kind"] == "submitted"]
    days = {frame["seq"]: frame["day"] for frame in frames}
    snapshots, subjects, seen_owned = {}, set(), set()
    displayed_observations = 0
    for frame in observations:
        obs = frame["observation"]
        seen_owned.update(entity["handle"] for name in ("heroes", "towns") for entity in obs[name])
        rows = _own_rows(frame)
        displayed_observations += bool(rows)
        for subject, row in rows.items():
            snapshots.setdefault(subject, []).append(_snapshot(frame, row))
            subjects.add(subject)
    final_seq = observations[-1]["seq"] if observations else None
    army_totals = []
    for subject, records in sorted(snapshots.items()):
        army_totals.append({"subject": subject, "observations": len(records), "first_observed": records[0],
            "maximum_observed": max(records, key=lambda row: row["creature_count"]),
            "last_observed": records[-1], "current_at_final_observation": records[-1]["seq"] == final_seq,
            "final_observation": records[-1] if records[-1]["seq"] == final_seq else None})

    recruits, army_actions = [], []
    for submitted in submissions:
        decision = decisions[submitted["decision_seq"]]
        action, before = submitted["action"], decision["observation"]
        result_frame = results.get(decision["seq"])
        result = result_frame["result"] if result_frame else None
        after_frame = None
        if result_frame:
            index = bisect_left(observation_seqs, result_frame["seq"]) - 1
            if index >= 0 and observations[index]["seq"] > decision["seq"]:
                after_frame = observations[index]
        old = _own_rows(decision)
        new = _own_rows(after_frame) if after_frame else {}
        if action["kind"] == "recruit":
            panel = (before.get("window") or {}).get("town_panel") or {}
            recruitment = panel.get("recruitment") or {}
            card = next((offer for offer in recruitment.get("offers", [])
                         if offer["ref"] == action.get("selection")), None)
            accepted = bool(result and result["accepted"] and result["status"] == "recruited")
            count = action.get("count")
            quantity = count if accepted and type(count) is int and card is not None else None
            # Destination refs route the native upper army, not necessarily its
            # entity subject. Match the displayed role rather than guessing IDs.
            role = recruitment.get("destination", {}).get("role")
            candidates = [subject for subject in old.keys() & new.keys()
                          if old[subject]["role"] == role and new[subject]["role"] == role]
            delta = None
            if card is not None and len(candidates) == 1:
                subject = candidates[0]
                delta = shared._counts(new[subject])[card["creature"]] - shared._counts(old[subject])[card["creature"]]
            recruits.append({"decision_seq": decision["seq"], "day": decision["day"],
                "result_seq": result_frame["seq"] if result_frame else None, "accepted": accepted,
                "result_status": result["status"] if result else None, "requested_quantity": count,
                "creature": card["creature"] if card else None, "accepted_quantity": quantity,
                "nonzero_success": accepted and type(count) is int and count > 0,
                "displayed_creature_delta": delta})
        if action["kind"] in shared.ARMY:
            effect = shared._army_effect(action, before, after_frame["observation"] if after_frame else None)
            comparable = sorted(old.keys() & new.keys())
            before_counts = [_snapshot(decision, old[subject]) for subject in comparable]
            after_counts = [_snapshot(after_frame, new[subject]) for subject in comparable]
            total_before = sum(row["creature_count"] for row in before_counts)
            total_after = sum(row["creature_count"] for row in after_counts)
            concentration = None
            if len(comparable) >= 2 and effect["transfer_scope"] == "across_armies":
                concentration = {"subjects": comparable, "total_before": total_before, "total_after": total_after,
                    "largest_count_share_before": max(row["creature_count"] for row in before_counts) / total_before if total_before else None,
                    "largest_count_share_after": max(row["creature_count"] for row in after_counts) / total_after if total_after else None}
            army_actions.append({"decision_seq": decision["seq"], "day": decision["day"], "kind": action["kind"],
                "result_status": result["status"] if result else None, "accepted": result["accepted"] if result else None,
                "before": before_counts, "after": after_counts, "effect": effect, "count_concentration": concentration})

    # Runs concern recorded submitted choices, not latent intent or efficiency.
    exact_repeats = same_kind = max_same_kind = window_run = max_window = army_run = max_army = 0
    previous_action = previous_kind = None
    for submitted in submissions:
        action, kind = submitted["action"], submitted["action"]["kind"]
        encoded = _canonical(action)
        exact_repeats += encoded == previous_action
        same_kind = same_kind + 1 if kind == previous_kind else 1
        max_same_kind = max(max_same_kind, same_kind)
        window_run = window_run + 1 if kind in shared.NAVIGATION else 0
        army_run = army_run + 1 if kind in shared.ARMY else 0
        max_window, max_army = max(max_window, window_run), max(max_army, army_run)
        previous_action, previous_kind = encoded, kind
    losses = [{"seq": change["seq"], "day": change["day"], "subject": subject}
              for change in basic["own_town_roster_changes"] for subject in change["removed"]]
    additions = [{"seq": change["seq"], "day": change["day"], "subject": subject}
                 for change in basic["own_town_roster_changes"] for subject in change["added"]]
    captures = [{**capture, "day": days[capture["seq"]]} for capture in basic["town_capture_evidence"]]
    return {"recruitment": {"attempts": len(recruits), "successful": sum(row["accepted"] for row in recruits),
        "successful_nonzero": sum(row["nonzero_success"] for row in recruits),
        "successful_zero": sum(row["accepted"] and row["requested_quantity"] == 0 for row in recruits),
        "quantity_observed_successes": sum(row["accepted_quantity"] is not None for row in recruits),
        "accepted_quantity": sum(row["accepted_quantity"] for row in recruits if row["accepted_quantity"] is not None),
        "displayed_delta_comparisons": sum(row["displayed_creature_delta"] is not None for row in recruits), "actions": recruits},
        "armies": {"coverage": {"observation_frames": len(observations), "displayed_army_observation_frames": displayed_observations,
            "owned_entity_refs_seen": sorted(seen_owned), "subjects_with_displayed_rows": sorted(subjects),
            "subjects_never_displayed": sorted(seen_owned - subjects), "whole_owned_army_census": False},
            "totals_by_subject": army_totals, "actions": army_actions},
        "towns": {"ownership_losses": losses, "roster_additions": additions, "corroborated_captures": captures},
        "repetition": {"submitted_choices": len(submissions), "consecutive_identical_action_repeats": exact_repeats,
            "maximum_consecutive_same_kind": max_same_kind, "maximum_consecutive_navigation_selection": max_window,
            "maximum_consecutive_army_operations": max_army},
        "terminal": next(({"seq": frame["seq"], "day": frame["day"], "outcome": frame["outcome"]}
                          for frame in frames if frame["kind"] == "terminal"), None)}


def per_episode(folder):
    """Read manifest-bound own observations; optionally verify saved behavior.json.

    Missing evidence is explicit incomplete coverage. Changed bindings or an
    existing behavior file inconsistent with independent reconstruction fail.
    This function has no native engine/model invocation and grants no admission.
    """
    folder = Path(folder).resolve()
    base = behavior.analyze_episode(folder)
    bindings = dict(base["bindings"])
    saved = folder / "behavior.json"
    if saved.is_file():
        data = saved.read_bytes()
        _require(_canonical(evidence.native.strict_json(data)) == _canonical(base), "Saved behavior differs from bound history")
        bindings["behavior.json"] = shared._binding(data)
    result = {"schema": SCHEMA, "episode_directory": str(folder), "learner_player": base["learner_player"],
        "episode": base.get("episode"), "bindings": bindings, "coverage": base["coverage"],
        "boundary": "unavailable", "behavior": base, "details": None, "native_replay_verified_by_this_module": False,
        "training_input": False, "policy_input": False, "reward_input": False, "training_samples": 0,
        "limitations": list(LIMITATIONS)}
    player = base["learner_player"]
    if player is None:
        return result
    entry = base["players"][evidence.COLORS[player]]
    if entry["metrics"] is not None:
        data = evidence.native.inside(folder, entry["history"]).read_bytes()
        _require(shared._binding(data) == bindings[entry["history"]], "History changed during metric reconstruction")
        lines = data.splitlines()
        # The existing parser already established exactly the complete prefix.
        frames = [evidence.native.strict_json(line) for line in lines[:entry["metrics"]["frames"]]]
        result["details"] = _extras(frames, entry["metrics"])
    episode = base["episode"]
    if base["coverage"]["complete"]:
        outcome = episode["outcomes_by_player"][evidence.COLORS[player]]
        result["boundary"] = "natural_" + outcome if episode["status"] == "completed" else "cutoff"
    elif episode["status"] == "failed":
        result["boundary"] = "failure"
    return result


def wilson(wins, natural_endings):
    """Two-sided 95% Wilson score interval for wins / natural endings only."""
    _require(type(wins) is int and type(natural_endings) is int and 0 <= wins <= natural_endings,
             "Wilson counts require integer 0 <= wins <= natural endings")
    if natural_endings == 0:
        return None
    z, n, p = 1.959963984540054, natural_endings, wins / natural_endings
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return {"confidence": .95, "lower": max(0., center-radius), "upper": min(1., center+radius)}


def aggregate(rows, *, phase, checkpoint):
    """Summarize scheduled evaluation attempts for one phase/checkpoint.

    Rows are runner records with episode, learner_player, engine_seed, status,
    and metrics=per_episode(folder) (or None). Passed status must already mean
    durable policy/native/cleanup verification by the caller. This is reporting,
    not a receipt constructor. Running rows remain explicitly unfinished.
    """
    rows = list(rows)
    _require(type(phase) is str and bool(phase) and checkpoint is not None, "Phase and checkpoint labels required")
    _canonical(checkpoint)
    seen, directories, counts = set(), set(), Counter()
    by_seat = {color: Counter() for color in evidence.COLORS}
    passed, observed, coverage = [], [], []
    for row in rows:
        episode, player = row["episode"], row["learner_player"]
        _require(type(episode) in (int, str) and type(player) is int and player in (0, 1), "Invalid scheduled episode/seat")
        _require(str(episode) not in seen, "Duplicate scheduled episode")
        seen.add(str(episode))
        if "phase" in row:
            _require(row["phase"] == phase, "Mixed evaluation phases")
        if "checkpoint" in row:
            _require(_canonical(row["checkpoint"]) == _canonical(checkpoint), "Mixed evaluation checkpoints")
        status = row["status"]
        _require(status in ("passed", "failed", "not_started", "running", "pending_verification"), "Unknown attempt status")
        metric = row.get("metrics")
        if metric is not None:
            _require(metric.get("schema") == SCHEMA, "Wrong metrics schema")
            _require(metric["episode_directory"] not in directories, "Duplicate captured episode")
            directories.add(metric["episode_directory"])
            if metric["learner_player"] is not None:
                _require(metric["learner_player"] == player and str(metric["episode"]["episode"]) == str(episode), "Capture differs from scheduled episode/seat")
                if metric["episode"].get("seed") is not None:
                    _require(_canonical(metric["episode"]["seed"]) == _canonical(row["engine_seed"]), "Capture differs from scheduled seed")
            observed.append(metric)
        if status == "passed":
            _require(metric is not None and metric["coverage"]["complete"] and metric["boundary"] in
                     ("natural_win", "natural_loss", "cutoff"), "Passed attempt lacks complete natural/cutoff metrics")
            category = metric["boundary"]
            passed.append(metric)
        else:
            category = {"failed": "failure", "not_started": "not_started", "running": "unfinished",
                        "pending_verification": "unfinished"}[status]
        counts[category] += 1
        by_seat[evidence.COLORS[player]][category] += 1
        coverage.append({"episode": episode, "learner_player": player, "engine_seed": row["engine_seed"],
                         "classification": category, "metrics_available": metric is not None,
                         "complete_history": bool(metric and metric["coverage"]["complete"])})

    def outcomes(counter):
        values = {key: counter[key] for key in ("natural_win", "natural_loss", "cutoff", "failure", "not_started", "unfinished")}
        natural = values["natural_win"] + values["natural_loss"]
        values.update(natural_endings=natural, attempted=sum(values.values())-values["not_started"],
            natural_win_rate=values["natural_win"] / natural if natural else None,
            natural_win_rate_wilson_95=wilson(values["natural_win"], natural))
        return values

    def descriptive(metrics):
        details = [metric["details"] for metric in metrics if metric["details"] is not None]
        recruitment = {key: sum(detail["recruitment"][key] for detail in details) for key in
            ("attempts", "successful", "successful_nonzero", "successful_zero", "quantity_observed_successes",
             "accepted_quantity", "displayed_delta_comparisons")}
        return {"basic": behavior.aggregate_episodes([metric["behavior"] for metric in metrics]),
            "recruitment": recruitment,
            "army_coverage": {"episodes_with_displayed_armies": sum(bool(detail["armies"]["totals_by_subject"]) for detail in details),
                "displayed_subjects_across_episodes": sum(len(detail["armies"]["totals_by_subject"]) for detail in details),
                "current_final_subjects_across_episodes": sum(row["current_at_final_observation"] for detail in details for row in detail["armies"]["totals_by_subject"])},
            "towns": {key: sum(len(detail["towns"][key]) for detail in details)
                      for key in ("ownership_losses", "roster_additions", "corroborated_captures")},
            "repetition": {"consecutive_identical_action_repeats": sum(detail["repetition"]["consecutive_identical_action_repeats"] for detail in details),
                "maximum_consecutive_navigation_selection": max((detail["repetition"]["maximum_consecutive_navigation_selection"] for detail in details), default=0),
                "maximum_consecutive_army_operations": max((detail["repetition"]["maximum_consecutive_army_operations"] for detail in details), default=0)}}

    return {"schema": AGGREGATE_SCHEMA, "phase": phase, "checkpoint": checkpoint, "scheduled": len(rows),
        "outcomes": outcomes(counts), "by_seat": {color: outcomes(count) for color, count in by_seat.items()},
        "attempts": coverage, "passed_attempt_metrics": descriptive(passed),
        "all_available_prefix_metrics": descriptive(observed), "training_input": False, "policy_input": False,
        "reward_input": False, "training_samples": 0, "best_checkpoint_selection": False,
        "limitations": list(LIMITATIONS)}
