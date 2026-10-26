"""Frozen own-player evaluation records against a typed native opponent.

No optimizer, opponent policy/value, paired trajectory or training targets are
constructed. The model still sees exactly the declared v7/v8 player-local stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from experiments.headless import builtin_evidence as evidence
from experiments.headless import contextual_evidence as native
from . import contextual as c
from .contextual_model import ModelConfig, model_schema_for, weights_sha256
from .contextual_rollout import (ContextualSession, BehaviorDecision, DecisionEstimate,
                                 RecurrentPrefix, _frame_bytes, _json_native, _wire_text)

SCHEMA = "heroes-builtin-evaluation-rollout-v1"
COLORS = evidence.COLORS


def _opponent_token(identity, player):
    evidence.require(type(identity) is dict and evidence.same(identity, evidence.opponent_identity(player, identity.get("binary_sha256"))),
                     "Expected actual typed Nullkiller identity")
    # The internal session's legacy argument is named opponent_checkpoint. This
    # tagged native library identity is not a neural checkpoint or model feature.
    return "builtin:Nullkiller:sha256:" + identity["binary_sha256"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BuiltinCollectedEpisode:
    history: tuple
    decisions: tuple
    estimates: tuple
    checkpoint: str
    weights_sha256: str
    policy_seed: int
    opponent_json: str
    manifest_json: str
    model_config_json: str

    purpose = "diagnostic"
    origin = "native"
    optimization_eligible = False
    training_samples = 0

    def __post_init__(self):
        evidence.require(type(self.history) is tuple and self.history, "A complete own history is required")
        evidence.require(type(self.decisions) is tuple and all(type(x) is BehaviorDecision for x in self.decisions)
                         and type(self.estimates) is tuple and all(type(x) is DecisionEstimate for x in self.estimates),
                         "Expected immutable actual own behavior records")
        c._integer(self.policy_seed, high=2**63 - 1)
        profile = evidence.manifest_profile(self.manifest)
        model = ModelConfig.from_dict(json.loads(self.model_config_json))
        evidence.require(model.schema == model_schema_for(profile.history_schema), "Model schema differs from own history")
        history = c.PlayerHistory(self.history[0].episode, self.history[0].player, schema=profile.history_schema)
        digest, prefixes = hashlib.sha256(), {}
        for frame in self.history:
            history.append(frame)
            digest.update(_frame_bytes(frame))
            if frame.kind == "decision":
                prefixes[frame.seq] = digest.hexdigest()
        evidence.validate_manifest_history(history, self.manifest)
        evidence.require(self.manifest["opponent_identity"] == self.opponent_identity, "Native opponent identity changed")
        token = _opponent_token(self.opponent_identity, self.player)
        wanted = tuple(f.seq for f in self.history if f.kind == "decision")
        evidence.require(tuple(x.decision_seq for x in self.estimates) == wanted, "Every own decision needs its actual value estimate")
        values = {}
        for item in self.estimates:
            expected = RecurrentPrefix(history.episode, self.player, self.checkpoint, self.weights_sha256,
                                       item.decision_seq, prefixes[item.decision_seq])
            evidence.require(item.prefix == expected, "Own value estimate has another recurrent prefix")
            values[item.decision_seq] = item.value
        records = {}
        for item in self.decisions:
            evidence.require(item.decision_seq in prefixes and item.decision_seq not in records,
                             "Behavior must bind a unique own decision")
            frame = self.history[item.decision_seq]
            expected = RecurrentPrefix(history.episode, self.player, self.checkpoint, self.weights_sha256,
                                       item.decision_seq, prefixes[item.decision_seq])
            evidence.require(item.prefix == expected and item.value == values[item.decision_seq]
                             and item.opponent_checkpoint == token and item.support.offers == frame.offers,
                             "Own behavior, support or native opponent binding changed")
            records[item.decision_seq] = item
        evidence.require(tuple(records) == tuple(sorted(records)), "Own choices must be chronological")
        evidence.require(all(f.decision_seq in records and records[f.decision_seq].action == f.action
                             for f in self.history if f.kind == "submitted"),
                             "A submitted action lacks actual own policy evidence")

    @property
    def player(self):
        return self.history[0].player

    @property
    def rollout(self):
        """Convenience own evaluation record; this is never a PPO rollout."""
        return self

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def opponent_identity(self):
        return json.loads(self.opponent_json)

    def to_dict(self):
        decisions = []
        for item in self.decisions:
            data = item.to_dict()
            del data["opponent_checkpoint"]
            decisions.append(data)
        return {"schema": SCHEMA, "player": self.player, "checkpoint": self.checkpoint,
            "weights_sha256": self.weights_sha256, "policy_seed": self.policy_seed,
            "opponent_identity": self.opponent_identity, "history": [c.encode_frame(f) for f in self.history],
            "decisions": decisions, "estimates": [x.to_dict() for x in self.estimates],
            "model_config": json.loads(self.model_config_json), "manifest": self.manifest,
            "purpose": self.purpose, "origin": self.origin, "optimization_eligible": False, "training_samples": 0}


def load_builtin_episode(path):
    raw = native.strict_json(Path(path).read_bytes()) if isinstance(path, (str, Path)) else path
    fields = {"schema", "player", "checkpoint", "weights_sha256", "policy_seed", "opponent_identity", "history",
              "decisions", "estimates", "model_config", "manifest", "purpose", "origin", "optimization_eligible", "training_samples"}
    evidence.require(type(raw) is dict and set(raw) == fields and raw["schema"] == SCHEMA
                     and raw["purpose"] == "diagnostic" and raw["origin"] == "native"
                     and raw["optimization_eligible"] is False and type(raw["training_samples"]) is int
                     and raw["training_samples"] == 0, "Expected an explicitly excluded own-player evaluation artifact")
    evidence.controllers(raw["player"])
    evidence.require(all(type(raw[k]) is list for k in ("history", "decisions", "estimates")), "Artifact arrays must be JSON arrays")
    frames = tuple(c.decode_frame(f) for f in raw["history"])
    evidence.require(frames and frames[0].player == raw["player"], "Artifact belongs to another player")
    token = _opponent_token(raw["opponent_identity"], raw["player"])
    decisions = []
    for item in raw["decisions"]:
        evidence.require(type(item) is dict and "opponent_checkpoint" not in item,
                         "A native opponent has no invented neural checkpoint")
        seq = item.get("decision_seq")
        evidence.require(type(seq) is int and 0 <= seq < len(frames) and frames[seq].kind == "decision",
                         "Behavior record references no actual own decision")
        decisions.append(BehaviorDecision.from_dict({**item, "opponent_checkpoint": token}, frames[seq]))
    estimates = []
    for item in raw["estimates"]:
        evidence.require(type(item) is dict and set(item) == {"decision_seq", "value", "prefix"}, "Malformed own value estimate")
        estimates.append(DecisionEstimate(item["decision_seq"], item["value"], RecurrentPrefix(**item["prefix"])))
    return BuiltinCollectedEpisode(frames, tuple(decisions), tuple(estimates), raw["checkpoint"],
        raw["weights_sha256"], raw["policy_seed"], _json(raw["opponent_identity"]), _json(raw["manifest"]), _json(raw["model_config"]))


def _delivered(records, typed):
    """The frames to consume for these delivered records.

    When the environment also hands over its own validated frames, and each
    delivered record has exact JSON types and exactly the canonical text of its
    frame, the session consumes those frames instead of decoding a fresh copy of
    each from its record. JSON text is injective over exact JSON types, so such
    a record is the canonical record of that frame, which the strict decoder
    maps back to an equal frame (canonical encoding round-trips); the model
    input is unchanged. Anything else, valid or not, takes the original path.
    """
    if (typed is None or type(typed) is not tuple or type(records) is not list or len(typed) != len(records)
            or not all(type(frame) is c.Frame and _wire_text(record) == c.canonical_frame(frame)
                       and _json_native(record) for record, frame in zip(records, typed))):
        return records
    return typed


class BuiltinCollector:
    """Exactly one unchanged frozen policy, own RNG and own recurrent history."""
    def __init__(self, model, checkpoint, *, learner_player, opponent_identity, policy_seed, interface=c.DIALOGS_ENVIRONMENT):
        self.profile = evidence.profile_for(interface)
        token = _opponent_token(opponent_identity, learner_player)
        evidence.require(model.config.schema == model_schema_for(self.profile.history_schema), "Frozen policy must match the explicit interface")
        c._integer(policy_seed, high=2**63 - 1)
        self.model, self.checkpoint, self.learner_player = model, checkpoint, learner_player
        self.opponent_identity, self.opponent_token = dict(opponent_identity), token
        self.policy_seed, self.session, self.sessions = policy_seed, None, None
        self._episodes = set()

    def start(self, episode):
        evidence.require(episode not in self._episodes and (self.session is None or self.session.finished),
                         "Use a fresh episode and finish the prior own session")
        self._episodes.add(episode)
        self.session = ContextualSession(self.model, episode, self.learner_player, self.checkpoint,
            self.opponent_token, schema=self.profile.history_schema, seed=self.policy_seed)
        self.sessions = (self.session,)

    def act(self, request):
        evidence.require(self.session is not None and not request.done and request.kind == "contextual"
            and request.player == self.learner_player and request.color == COLORS[self.learner_player]
            and not request.past_horizon and request.reward == 0 and request.frames,
            "Only the current own decision can be sampled")
        # consume decoded request.frames[-1] (or matched its exact accepted text); the
        # record is normally that same object, and is decoded only when it is not.
        last = self.session.consume(_delivered(request.frames, getattr(request, "typed_frames", None)))
        evidence.require(last.kind == "decision" and (request.record is request.frames[-1]
                                                      or c.decode_frame(request.record) == last),
                         "Request record and actual own decision differ")
        return self.session.choose(last.seq).action

    def finish(self, end, *, manifest_path):
        evidence.require(self.session is not None and end.done, "An actual episode boundary is required")
        color = COLORS[self.learner_player]
        evidence.require(set(end.final_frames) == {color}, "A native opponent cannot acquire a policy stream")
        self.session.consume(_delivered(end.final_frames[color], getattr(end, "final_typed_frames", {}).get(color)))
        manifest = native.strict_json(Path(manifest_path).read_bytes())
        evidence.require(manifest == end.manifest and manifest["opponent_identity"] == self.opponent_identity,
                         "Persisted episode or native opponent changed")
        evidence.validate_manifest_history(self.session.history, manifest)
        evidence.read_finalization(Path(manifest_path).parent, manifest)
        declaration = manifest["players"][color][self.profile.history_key]
        path = native.inside(Path(manifest_path).parent, declaration["path"])
        data = path.read_bytes()
        evidence.require(hashlib.sha256(data).hexdigest() == declaration["sha256"]
                         and data == b"".join(_frame_bytes(f) for f in self.session.history.frames),
                         "Persisted own history differs from model experience")
        result = BuiltinCollectedEpisode(self.session.history.frames, tuple(self.session._choices.values()),
            tuple(DecisionEstimate(seq, value, self.session._prefixes[seq]) for seq, value in self.session._values.items()),
            self.checkpoint, self.session.weights_sha256, self.policy_seed, _json(self.opponent_identity),
            _json(manifest), _json(self.model.config.to_dict()))
        self.session._check_weights()
        self.session.finished = True
        return result

    def run_episode(self, env, *, episode, seed, max_requests=2040):
        evidence.require(env.interface == self.profile.environment and env.learner_player == self.learner_player,
                         "Collector requires its own single-player environment")
        c._integer(max_requests, low=1)
        self.start(str(episode))
        request, choices = env.reset(episode=episode, seed=seed), 0
        while not request.done:
            if choices >= max_requests:
                if env.sides[self.learner_player].validator.live.requires_choice:
                    env.defer_dialog_cutoff("collection_window")
                    request = env.step(self.act(request))
                    choices += 1
                else:
                    # The environment already delivered these frames. Retain
                    # their memory and actual safe-decision value, without
                    # advancing the sampling RNG or choosing another action.
                    self.session.consume(_delivered(request.frames, getattr(request, "typed_frames", None)))
                    request = env.truncate("collection_window")
            else:
                request = env.step(self.act(request))
                choices += 1
        return self.finish(request, manifest_path=env.episode_dir / "manifest.json")


def audit_persisted_builtin_policy(folder, model, checkpoint, policy_seed, opponent_identity):
    """Independent own-memory/RNG reconstruction, never an opponent policy claim."""
    from validation.probes.validate_hero_policy import reconstruct_behavior
    folder = Path(folder)
    collected = load_builtin_episode(folder / "builtin-rollout.json")
    manifest = native.strict_json((folder / "manifest.json").read_bytes())
    evidence.require(collected.manifest == manifest and collected.checkpoint == checkpoint
                     and collected.policy_seed == policy_seed and collected.opponent_identity == opponent_identity
                     and collected.weights_sha256 == weights_sha256(model)
                     and json.loads(collected.model_config_json) == model.config.to_dict(),
                     "Saved own policy or episode differs from the frozen reconstruction inputs")
    profile = evidence.manifest_profile(manifest)
    history = c.PlayerHistory(collected.history[0].episode, collected.player, schema=profile.history_schema)
    for frame in collected.history:
        history.append(frame)
    audit = evidence.validate_manifest_history(history, manifest)
    evidence.read_finalization(folder, manifest)
    declaration = manifest["players"][COLORS[collected.player]][profile.history_key]
    data = native.inside(folder, declaration["path"]).read_bytes()
    evidence.require(hashlib.sha256(data).hexdigest() == declaration["sha256"]
                     and data == b"".join(_frame_bytes(f) for f in collected.history),
                     "Artifact omitted or changed the persisted own experience")
    rebuilt = reconstruct_behavior(collected.history, collected.decisions, collected.estimates,
                                   model, policy_seed, schema=profile.history_schema)
    return {"passed": True, "player": collected.player, "history_audit": audit, **rebuilt}


def retain_failed_builtin_evidence(env, collector, attempt, error, *, cleanup_seconds=30.0):
    """Persist an actual technical failure; never answer a dialog or invent cutoff."""
    from validation.probes.validate_hero_policy import wall_limit
    from experiments.episodes.scoring import write_json
    if env is None:
        return
    folder = env.episode_dir
    if folder is not None:
        attempt["episode_directory"] = str(folder)
    if env.process is not None and env.result is None:
        try:
            with wall_limit(cleanup_seconds):
                env._note_problem("watchdog_timeout" if isinstance(error, TimeoutError) else "protocol_error",
                                  "Frozen built-in evaluation failed: " + attempt.get("error", str(error)))
                end = env._finish(None)
                evidence.require(end.status == "failed" and end.outcomes == dict.fromkeys(COLORS)
                                 and end.rewards == dict.fromkeys(COLORS), "Failed cleanup produced a score")
                attempt.update(episode_status=end.status, termination_reason=end.termination_reason,
                               failure_finalization="persisted")
        except Exception as secondary:
            attempt["failure_finalization_error"] = f"{type(secondary).__name__}: {secondary}"
    if folder is not None and collector is not None and collector.session is not None:
        try:
            session = collector.session
            choices = [item.to_dict() for item in session._choices.values()]
            for item in choices:
                del item["opponent_checkpoint"]
            write_json(folder / "partial-policy-evidence.json", {"incomplete": True, "training_input": False,
                "player": session.history.player, "opponent_identity": collector.opponent_identity,
                "decisions": choices, "history": [c.encode_frame(f) for f in session.history.frames]})
        except Exception as secondary:
            attempt["partial_evidence_error"] = f"{type(secondary).__name__}: {secondary}"
