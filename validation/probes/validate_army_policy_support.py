"""Offline counterfactual scoring of native-validated operator-controlled v6/v7 history.

The recorded actions were chosen by the diagnostic operator, not sampled from
this checkpoint. This tool checks their support and probabilities under frozen
weights, including all intervening player-visible frames. It does not execute
actions, generate demonstrations, collect rollout samples, or admit training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.headless import contextual_evidence as evidence
from experiments.headless.contextual_replay import REPORT_SCHEMA, validate_native_report
from experiments.headless.contextual_replay_history import verify_histories
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint
from experiments.learner.contextual_model import ContextualEncoder, MANAGEMENT_MODEL_SCHEMA, DIALOGS_MODEL_SCHEMA, model_schema_for, weights_sha256

COLORS = ("red", "blue")
SCHEMA = "heroes-v6-counterfactual-policy-support-v1"
DIALOGS_SCHEMA = "heroes-v7-counterfactual-policy-support-v1"
V6_ACTIONS = ("split_stack", "merge_stacks", "swap_stacks", "open_stack",
              "select_upgrade", "upgrade_stack", "swap_garrison")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    header = canonical({"dtype": str(value.dtype), "shape": list(value.shape)}).encode()
    return hashlib.sha256(header + b"\n" + bytes(value.reshape(-1).view(torch.uint8).tolist())).hexdigest()


def assert_frozen(model):
    require(model.config.schema in (MANAGEMENT_MODEL_SCHEMA, DIALOGS_MODEL_SCHEMA), "Only explicit v6/v7 models are supported")
    require(not model.training and all(not parameter.requires_grad and parameter.device.type == "cpu"
                                      for parameter in model.parameters()), "Model must be frozen on CPU")


def score_history(frames, model):
    """One fresh player-local fold; a submitted action is evaluated, never sampled.

    The held decision distribution is retained until that player's actual
    submitted frame appears. No future action or other player's history enters
    the recurrent fold. Technical metadata frames are checked but intentionally
    leave memory unchanged, according to the production encoder.
    """
    frames = tuple(frames)
    require(frames and frames[0].schema in (c.MANAGEMENT_SCHEMA, c.DIALOGS_SCHEMA)
            and all(frame.schema == frames[0].schema for frame in frames), "Expected nonempty v6/v7 history")
    profile = c.profile_for(frames[0].schema)
    assert_frozen(model)
    require(model.config.schema == model_schema_for(profile.environment), "Model/history profile mismatch")
    before = weights_sha256(model)
    encoder = ContextualEncoder(frames[0].episode, frames[0].player, profile.history_schema)
    state = model.initial_state()
    initial_memory = tensor_digest(state)
    records, decisions = [], []
    held = None
    with torch.no_grad():
        for frame in frames:
            encoded = encoder.encode(frame)
            if frame.kind == "submitted":
                require(held is not None and held[0] == frame.decision_seq, "Submitted action has no held distribution")
                _, distribution, record = held
                require(record["operator_action"] is None and distribution.support.offers.permits(frame.action),
                        "Repeated or unsupported operator choice")
                evaluated = distribution.evaluate(frame.action)
                joint = evaluated.log_prob.item()
                require(math.isfinite(joint), "Non-finite counterfactual joint probability")
                trace = [{"name": part.name, "selected_index": part.selected_index,
                          "support_size": part.support_size, "log_prob": part.log_prob.item()}
                         for part in evaluated.conditionals]
                require(trace and all(math.isfinite(part["log_prob"]) and 0 <= part["selected_index"] < part["support_size"]
                                      for part in trace), "Invalid conditional probability trace")
                # Use the model's tensor arithmetic to avoid different Python
                # float accumulation rounding when checking its exact joint.
                require(torch.stack([part.log_prob for part in evaluated.conditionals]).sum().item() == joint,
                        "Joint probability differs from its conditional sum")
                record.update(submitted_seq=frame.seq, operator_action=asdict(frame.action),
                              joint_log_prob=joint, conditionals=trace)
            step = model.forward_step(encoded, state)
            state = step.state
            records.append({"seq": frame.seq, "kind": frame.kind, "encoded": encoded.semantics is not None,
                            "frame_sha256": hashlib.sha256(canonical(c.encode_frame(frame)).encode()).hexdigest(),
                            "memory_sha256": tensor_digest(state)})
            if frame.kind == "decision":
                require(step.decision is not None, "Decision has no policy/value output")
                value = step.decision.value.item()
                require(math.isfinite(value), "Non-finite counterfactual value")
                distribution = step.decision.distribution
                require(distribution.support.offers == frame.offers and distribution.support.context == frame.observation.context,
                        "Model support differs from the exact recorded decision")
                support = json.loads(canonical(asdict(distribution.support)))
                record = {"decision_seq": frame.seq, "context": asdict(frame.observation.context),
                          "value": value, "support": support,
                          "support_sha256": hashlib.sha256(canonical(support).encode()).hexdigest(),
                          "submitted_seq": None, "operator_action": None,
                          "joint_log_prob": None, "conditionals": []}
                decisions.append(record)
                held = (frame.seq, distribution, record)
            else:
                require(step.decision is None, "A non-decision frame unexpectedly produced a decision")
    require(weights_sha256(model) == before, "Counterfactual scoring changed weights")
    actions = Counter(row["operator_action"]["kind"] for row in decisions if row["operator_action"] is not None)
    return {"schema": DIALOGS_SCHEMA if profile.dialogs else SCHEMA, "scope": "counterfactual_scoring_of_recorded_operator_choices",
            "episode": frames[0].episode, "player": frames[0].player,
            "checkpoint_weights_sha256": before, "initial_memory_sha256": initial_memory,
            "frames": records, "decisions": decisions, "action_kinds_scored": dict(sorted(actions.items())),
            "frames_folded": len(records), "values_scored": len(decisions), "operator_actions_scored": sum(actions.values()),
            "conditionals_scored": sum(len(row["conditionals"]) for row in decisions),
            "weights_unchanged": True, "private_memory_reset": True,
            "actions_sampled": 0, "actions_executed": 0, "optimization_eligible": False,
            "training_samples": 0, "optimizer_updates": 0}


def compare_folds(first, second):
    require(canonical(first) == canonical(second), "Independent player-memory folds differ")
    return {"exact": True, "maximum_absolute_difference": 0,
            "frames_compared": first["frames_folded"], "values_compared": first["values_scored"],
            "supports_compared": first["values_scored"], "joint_probabilities_compared": first["operator_actions_scored"],
            "conditional_probabilities_compared": first["conditionals_scored"]}


def validate_replay_binding(manifest_path, manifest, replay):
    require(type(replay) is dict and replay.get("schema") == REPORT_SCHEMA
            and replay.get("kind") == "contextual-native-replay" and replay.get("verified") is True
            and replay.get("episode") == manifest.get("episode")
            and replay.get("training_performed") is False, "A successful independent native replay report is required")
    require(replay.get("inputs", {}).get("manifest") == evidence.file_identity(manifest_path),
            "Replay report is bound to another manifest")
    require(manifest.get("environment") in (c.MANAGEMENT_ENVIRONMENT, c.DIALOGS_ENVIRONMENT),
            "Only native-validated v6/v7 natural/cutoff history is supported")
    profile = c.profile_for(manifest["environment"])
    require(manifest.get("history_schema") == profile.history_schema
            and manifest.get("action_schema") == profile.action_schema
            and manifest.get("status") in ("completed", "truncated"), "Only native-validated v6/v7 natural/cutoff history is supported")


def load_inputs(episode, replay_path):
    episode, replay_path = Path(episode).resolve(), Path(replay_path).resolve()
    manifest_path = episode / "manifest.json"
    manifest = evidence.strict_json(manifest_path.read_bytes())
    replay = evidence.strict_json(replay_path.read_bytes())
    validate_replay_binding(manifest_path, manifest, replay)
    final = evidence.strict_json((episode / "contextual-finalization.json").read_bytes())
    reconstructed = verify_histories(episode, manifest, final)
    require(reconstructed == replay.get("histories") and reconstructed.get("verified") is True,
            "Saved replay and independently reconstructed histories disagree")
    validate_native_report(replay.get("native", {}), replay.get("wire", {}), final,
                           cutoff=manifest["status"] == "truncated")
    replay_files = manifest.get("replay", {}).get("files", {})
    captured = evidence.capture_files(episode)
    require(captured["recorded"] and captured["files"] == replay_files,
            "Native replay inputs or terminal evidence changed after finalization")
    bindings = {"manifest": {"path": str(manifest_path), **evidence.file_identity(manifest_path)},
                "replay_report": {"path": str(replay_path), **evidence.file_identity(replay_path)},
                "finalization": {"path": str(episode / "contextual-finalization.json"),
                                 **evidence.file_identity(episode / "contextual-finalization.json")}}
    for name, declared in replay_files.items():
        path = evidence.inside(episode, name)
        require(evidence.file_identity(path) == declared, "Recorded native replay input changed: " + name)
        bindings[name] = {"path": str(path), **declared}
    histories = []
    for player, color in enumerate(COLORS):
        declaration = manifest["players"][color][c.profile_for(manifest["environment"]).history_key]
        path = evidence.inside(episode, declaration["path"])
        identity = evidence.file_identity(path)
        require(identity["sha256"] == declaration["sha256"], "Player history differs from its manifest")
        frames = tuple(c.decode_frame(evidence.strict_json(line)) for line in path.read_bytes().splitlines())
        require(frames and all(frame.player == player and frame.episode == manifest["episode"] for frame in frames),
                "History contains another player or episode")
        histories.append(frames)
        bindings[color + "_history"] = {"path": str(path), **identity,
                                         "canonical_sha256": declaration["canonical_sha256"]}
        for suffix, relative in (("bridge", f"agents/{color}/bridge.jsonl"),
                                 ("live", manifest["players"][color]["live_evidence"]["path"])):
            path = evidence.inside(episode, relative)
            bindings[color + "_" + suffix] = {"path": str(path), **evidence.file_identity(path)}
    return manifest, tuple(histories), bindings, reconstructed


def check_bindings(bindings):
    for name, bound in bindings.items():
        require(evidence.file_identity(Path(bound["path"])) == {key: bound[key] for key in ("bytes", "sha256")},
                "Bound input changed during scoring: " + name)


def run(args):
    output = Path(args.output).expanduser().resolve()
    require(not output.exists(), "Output already exists; preserve prior attempts")
    output.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "running", "scope": "counterfactual_scoring_of_recorded_operator_choices",
              "actions_were_policy_sampled": False, "native_action_execution": False,
              "optimization_eligible": False, "training_samples": 0, "optimizer_updates": 0,
              "training_performed": False, "players": [],
              "requested_inputs": {name: str(Path(getattr(args, name)).expanduser().resolve())
                                   for name in ("episode", "replay_report", "checkpoint")}}
    write_json(output / "report.json", report, exclusive=True)
    try:
        torch.set_num_threads(1)
        manifest, histories, bindings, history_proof = load_inputs(args.episode, args.replay_report)
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        bindings["checkpoint"] = {"path": str(checkpoint), **evidence.file_identity(checkpoint)}
        sources = (Path(__file__), Path(c.__file__), ROOT / "experiments/learner/contextual_model.py",
                   ROOT / "experiments/learner/checkpoints.py", ROOT / "experiments/headless/contextual_replay_history.py",
                   ROOT / "experiments/headless/contextual_protocol.py", ROOT / "experiments/headless/contextual_replay.py",
                   Path(evidence.__file__))
        for path in sources:
            bindings["scoring_source:" + str(path.relative_to(ROOT))] = {"path": str(path), **evidence.file_identity(path)}
        profile = c.profile_for(manifest["environment"])
        report["schema"] = DIALOGS_SCHEMA if profile.dialogs else SCHEMA
        loaded = load_policy_checkpoint(checkpoint, interface=profile.environment)
        loaded.model.requires_grad_(False)
        assert_frozen(loaded.model)
        report.update(episode=manifest["episode"], native_boundary=manifest["status"], bindings=bindings,
                      checkpoint={"id": loaded.checkpoint, "weights_sha256": loaded.weights_sha256,
                                  "provenance": loaded.provenance, "source_metadata": loaded.source_metadata},
                      history_reconstruction=history_proof)
        write_json(output / "report.json", report)
        all_actions = Counter()
        # The second pass reverses player order. No encoder or GRU state is
        # shared between seats or folds, even though their frozen weights are.
        for player, frames in enumerate(histories):
            check_bindings(bindings)
            first = score_history(frames, loaded.model)
            path = output / (COLORS[player] + "-fold-1.json")
            write_json(path, first, exclusive=True)
            all_actions.update(first["action_kinds_scored"])
            report["players"].append({"player": player, "fold_1": evidence.file_identity(path),
                                      "frames": first["frames_folded"], "operator_actions_scored": first["operator_actions_scored"],
                                      "action_kinds_scored": first["action_kinds_scored"]})
            write_json(output / "report.json", report)
        for player in (1, 0):
            check_bindings(bindings)
            second = score_history(histories[player], loaded.model)
            path = output / (COLORS[player] + "-fold-2.json")
            write_json(path, second, exclusive=True)
            first = evidence.strict_json((output / (COLORS[player] + "-fold-1.json")).read_bytes())
            report["players"][player].update(fold_2=evidence.file_identity(path), comparison=compare_folds(first, second))
            write_json(output / "report.json", report)
        check_bindings(bindings)
        require(weights_sha256(loaded.model) == loaded.weights_sha256, "Checkpoint weights changed during scoring")
        required = ("choose_dialog",) if profile.dialogs else V6_ACTIONS
        missing = sorted(set(required) - all_actions.keys())
        coverage = ({"missing_v7_action_kinds": missing, "all_v7_action_kinds_covered": not missing}
                    if profile.dialogs else {"missing_v6_action_kinds": missing, "all_v6_action_kinds_covered": not missing})
        report.update(status="passed", action_kinds_scored=dict(sorted(all_actions.items())), **coverage,
                      coverage_scope="Only operator actions present in this bound episode; absent branches are not claimed",
                      independent_player_memories=True, independent_memory_folds=2, weights_unchanged=True,
                      maximum_absolute_difference=0)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "replay-report", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({"status": report["status"], "scope": report["scope"], "training_samples": 0,
                      "missing_v6_action_kinds": report.get("missing_v6_action_kinds"), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
