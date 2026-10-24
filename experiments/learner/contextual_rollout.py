"""Fixed-weight contextual collection; no optimizer or engine outcome inference.

Behavior choices become samples only after consumption and an ordinary result
or an explicitly versioned verified natural terminal closure. Both complete
histories and finalization proofs are checked before eligibility. Diagnostic
native runs never become PPO data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import torch

from . import contextual as c
from .contextual_model import ContextualEncoder, DecisionSupport, ModelConfig, distribution_schema_for, model_schema_for, weights_sha256

ROLLOUT_SCHEMA = "heroes-context-rollout-v1"
TERMINAL_ROLLOUT_SCHEMA = "heroes-context-rollout-v2"
_TRAJECTORIES = {ROLLOUT_SCHEMA: c.TRAJECTORY_SCHEMA, TERMINAL_ROLLOUT_SCHEMA: c.TERMINAL_TRAJECTORY_SCHEMA}
COLORS = ("red", "blue")


def _text(value, name):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"Invalid {name}")


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("Expected a SHA-256 digest")


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")


def _tuple(value, cls):
    if type(value) is not tuple or any(type(item) is not cls for item in value):
        raise ValueError(f"Expected immutable tuple of {cls.__name__}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _frame_bytes(frame):
    return (_canonical(c.encode_frame(frame)) + "\n").encode()


def _support_dict(support):
    offers = c.encode_offers(support.offers)
    return {"action_schema": support.action_schema, "context": offers["context"], "offers": offers,
            "distribution_schema": support.distribution_schema}


def _support(frame):
    return DecisionSupport(action_schema=frame.offers.schema, context=frame.offers.context, offers=frame.offers,
                           distribution_schema=distribution_schema_for(frame.offers.schema))


def _record(raw, keys):
    if type(raw) is not dict or set(raw) != set(keys):
        raise ValueError("Unexpected or missing rollout record fields")
    return dict(raw)


@dataclass(frozen=True)
class RecurrentPrefix:
    episode: str
    player: int
    checkpoint: str
    weights_sha256: str
    through_seq: int
    history_sha256: str

    def __post_init__(self):
        _text(self.episode, "episode"); _text(self.checkpoint, "checkpoint")
        c._integer(self.player, high=1); c._integer(self.through_seq)
        _hash(self.weights_sha256); _hash(self.history_sha256)

    def to_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class ConditionalRecord:
    name: str
    selected_index: int
    support_size: int
    log_prob: float

    def __post_init__(self):
        _text(self.name, "conditional name")
        c._integer(self.support_size, low=1)
        c._integer(self.selected_index, high=self.support_size - 1)
        _finite(self.log_prob, "conditional probability")
        if self.log_prob > 1e-6:
            raise ValueError("Conditional log probability must not be positive")


@dataclass(frozen=True)
class DecisionEstimate:
    """A value at an actual own decision, even when no action is sampled."""
    decision_seq: int
    value: float
    prefix: RecurrentPrefix

    def __post_init__(self):
        c._integer(self.decision_seq); _finite(self.value, "decision value")
        if type(self.prefix) is not RecurrentPrefix or self.prefix.through_seq != self.decision_seq:
            raise ValueError("Decision estimate requires its exact recurrent prefix")

    def to_dict(self):
        return {"decision_seq": self.decision_seq, "value": self.value, "prefix": self.prefix.to_dict()}


@dataclass(frozen=True)
class BehaviorDecision:
    decision_seq: int
    action: c.Action
    support: DecisionSupport
    log_prob: float
    value: float
    conditionals: tuple[ConditionalRecord, ...]
    prefix: RecurrentPrefix
    opponent_checkpoint: str

    def __post_init__(self):
        c._integer(self.decision_seq)
        if type(self.action) is not c.Action or type(self.support) is not DecisionSupport:
            raise ValueError("Behavior requires typed action and exact support")
        if not self.support.offers.permits(self.action):
            raise ValueError("Behavior action is outside its recorded support")
        _finite(self.log_prob, "behavior log probability"); _finite(self.value, "behavior value")
        _tuple(self.conditionals, ConditionalRecord)
        if (not self.conditionals or self.log_prob > 1e-6
                or not math.isclose(self.log_prob, sum(item.log_prob for item in self.conditionals), abs_tol=2e-5, rel_tol=2e-5)):
            raise ValueError("Joint probability must bind every recorded conditional")
        if type(self.prefix) is not RecurrentPrefix or self.prefix.through_seq != self.decision_seq:
            raise ValueError("Behavior prefix must end at its own decision")
        _text(self.opponent_checkpoint, "frozen opponent checkpoint")

    def to_dict(self):
        return {"decision_seq": self.decision_seq, "action": c.encode_action(self.action),
                "support": _support_dict(self.support), "log_prob": self.log_prob, "value": self.value,
                "conditionals": [dict(vars(item)) for item in self.conditionals],
                "prefix": self.prefix.to_dict(), "opponent_checkpoint": self.opponent_checkpoint}

    @classmethod
    def from_dict(cls, raw, frame):
        data = _record(raw, ("decision_seq", "action", "support", "log_prob", "value", "conditionals", "prefix", "opponent_checkpoint"))
        support = _support(frame)
        if data["support"] != _support_dict(support):
            raise ValueError("Serialized support disagrees with the actual decision")
        action = c.decode_frame(dict(schema=frame.schema, episode=frame.episode, player=frame.player,
            seq=frame.seq + 1, day=frame.day, kind="submitted", decision_seq=frame.seq, action=data["action"])).action
        if type(data["conditionals"]) is not list:
            raise ValueError("Conditional records require a JSON array")
        try:
            return cls(data["decision_seq"], action, support, data["log_prob"], data["value"],
                       tuple(ConditionalRecord(**item) for item in data["conditionals"]),
                       RecurrentPrefix(**data["prefix"]), data["opponent_checkpoint"])
        except TypeError as error:
            raise ValueError("Invalid serialized behavior record") from error


@dataclass(frozen=True)
class RolloutSample:
    decision: BehaviorDecision
    result_seq: int | None
    successor_seq: int | None
    reward: float | None
    bootstrap_value: float | None
    eligible: bool
    terminal_closure: c.TerminalClosure | None = None

    def __post_init__(self):
        if type(self.decision) is not BehaviorDecision or type(self.eligible) is not bool:
            raise ValueError("Sample requires immutable behavior and explicit eligibility")
        if self.terminal_closure is None:
            c._integer(self.result_seq, low=self.decision.decision_seq + 1)
            if self.successor_seq is not None: c._integer(self.successor_seq, low=self.result_seq + 1)
        elif (type(self.terminal_closure) is not c.TerminalClosure or self.result_seq is not None
                or self.terminal_closure.consumed_seq <= self.decision.decision_seq
                or self.successor_seq is not None or self.bootstrap_value is not None
                or self.reward not in (-1, 1) or not self.eligible):
            raise ValueError("Terminal closure has only verified terminal reward, never an ordinary result/bootstrap")
        if self.reward is not None and (type(self.reward) not in (int, float) or self.reward not in (-1, 0, 1)):
            raise ValueError("Only terminal win/loss and ordinary zero rewards are permitted")
        if self.bootstrap_value is not None: _finite(self.bootstrap_value, "bootstrap value")
        if self.eligible and (self.reward is None or (self.successor_seq is not None) != (self.bootstrap_value is not None)):
            raise ValueError("Eligible samples require their actual same-player successor value")

    def to_dict(self, *, schema=c.TRAJECTORY_SCHEMA):
        if schema not in _TRAJECTORIES.values() or schema == c.TRAJECTORY_SCHEMA and self.terminal_closure is not None:
            raise ValueError("Terminal closure requires trajectory v2")
        result = {"decision_seq": self.decision.decision_seq, "result_seq": self.result_seq,
                "successor_seq": self.successor_seq, "reward": self.reward,
                "bootstrap_value": self.bootstrap_value, "eligible": self.eligible}
        if schema == c.TERMINAL_TRAJECTORY_SCHEMA:
            result["terminal_closure"] = None if self.terminal_closure is None else dict(vars(self.terminal_closure))
        return result


@dataclass(frozen=True)
class ContextualRollout:
    episode: str
    player: int
    role: str
    history: tuple[c.Frame, ...]
    decisions: tuple[BehaviorDecision, ...]
    samples: tuple[RolloutSample, ...]
    boundary: str
    history_sha256: str
    evidence_ref: str
    behavior_checkpoint: str
    opponent_checkpoint: str
    weights_sha256: str
    opponent_weights_sha256: str
    opponent_pool: str
    purpose: str
    origin: str
    estimates: tuple[DecisionEstimate, ...] = ()
    trajectory_schema: str = c.TRAJECTORY_SCHEMA
    _optimization_eligible: bool = field(init=False, repr=False, compare=False)
    _trajectory_eligible: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _text(self.episode, "episode"); c._integer(self.player, high=1)
        if self.trajectory_schema not in _TRAJECTORIES.values(): raise ValueError("Unsupported trajectory schema")
        for value in (self.behavior_checkpoint, self.opponent_checkpoint, self.opponent_pool, self.evidence_ref): _text(value, "provenance")
        for value in (self.history_sha256, self.weights_sha256, self.opponent_weights_sha256): _hash(value)
        if (self.role not in ("learner", "frozen_opponent") or self.purpose not in ("diagnostic", "synthetic", "training")
                or self.origin not in ("native", "synthetic") or self.purpose == "synthetic" and self.origin != "synthetic"):
            raise ValueError("Invalid collection role or purpose/origin")
        _tuple(self.history, c.Frame); _tuple(self.decisions, BehaviorDecision); _tuple(self.samples, RolloutSample)
        _tuple(self.estimates, DecisionEstimate)
        if not self.history: raise ValueError("Rollout requires complete player history")
        history = c.PlayerHistory(self.episode, self.player, schema=self.history[0].schema)
        digest = hashlib.sha256(); prefixes = {}
        for frame in self.history:
            history.append(frame); digest.update(_frame_bytes(frame)); prefixes[frame.seq] = digest.hexdigest()
        if history.digest() != self.history_sha256 or history.boundary != self.boundary or self.boundary not in ("natural", "cutoff", "failure"):
            raise ValueError("Rollout history hash/boundary mismatch")
        expected_sequences = tuple(frame.seq for frame in self.history if frame.kind == "decision")
        if tuple(item.decision_seq for item in self.estimates) != expected_sequences:
            raise ValueError("Every actual own decision requires its fixed-weight value estimate")
        estimates = {}
        for item in self.estimates:
            expected = RecurrentPrefix(self.episode, self.player, self.behavior_checkpoint, self.weights_sha256,
                                       item.decision_seq, prefixes[item.decision_seq])
            if item.prefix != expected: raise ValueError("Decision value prefix is not its complete own history")
            estimates[item.decision_seq] = item
        records = {}
        for record in self.decisions:
            seq = record.decision_seq
            if seq in records or seq >= len(self.history) or self.history[seq].kind != "decision":
                raise ValueError("Behavior must reference unique actual player decisions")
            if record.support != _support(self.history[seq]) or record.opponent_checkpoint != self.opponent_checkpoint:
                raise ValueError("Behavior support/opponent changed")
            expected = RecurrentPrefix(self.episode, self.player, self.behavior_checkpoint, self.weights_sha256, seq, prefixes[seq])
            if record.prefix != expected: raise ValueError("Behavior recurrent prefix is not the complete own history")
            if record.value != estimates[seq].value: raise ValueError("Sampled value differs from its own decision estimate")
            records[seq] = record
        if tuple(records) != tuple(sorted(records)): raise ValueError("Behavior decisions must be chronological")
        for frame in self.history:
            if frame.kind == "submitted" and (frame.decision_seq not in records or records[frame.decision_seq].action != frame.action):
                raise ValueError("Every submitted action requires matching fixed behavior evidence")
        transitions = history.transitions(schema=self.trajectory_schema)
        if len(self.samples) != len(transitions): raise ValueError("Samples must exactly match resolved contextual transitions")
        for sample, transition in zip(self.samples, transitions):
            if (sample.decision != records.get(transition.decision_seq) or sample.result_seq != transition.result_seq
                    or sample.successor_seq != transition.successor_seq or sample.reward != transition.reward
                    or sample.eligible != transition.eligible or sample.terminal_closure != transition.terminal_closure):
                raise ValueError("Sample disagrees with the strict contextual trajectory")
            expected_bootstrap = estimates[transition.successor_seq].value if transition.bootstrap_allowed else None
            if sample.bootstrap_value != expected_bootstrap:
                raise ValueError("Sample successor value is not its actual same-player fixed-weight estimate")
        consumed = {f.decision_seq for f in self.history if f.kind == "consumed"}
        completed = {f.decision_seq for f in self.history if f.kind == "result"}
        completed.update(s.decision.decision_seq for s in self.samples if s.terminal_closure is not None)
        contiguous = bool(self.samples) and consumed == completed and all(sample.eligible for sample in self.samples)
        for first, second in zip(self.samples, self.samples[1:]):
            contiguous = contiguous and first.successor_seq == second.decision.decision_seq
        if self.samples:
            last = self.samples[-1]
            if self.boundary == "natural":
                contiguous = contiguous and last.successor_seq is None and last.bootstrap_value is None and last.reward in (-1, 1)
            elif self.boundary == "cutoff":
                cutoff = next((frame for frame in self.history if frame.kind == "cutoff"), None)
                contiguous = contiguous and cutoff is not None and last.successor_seq == cutoff.decision_seq and last.bootstrap_value is not None
            else: contiguous = False
        object.__setattr__(self, "_trajectory_eligible", bool(contiguous and self.boundary in ("natural", "cutoff")))
        object.__setattr__(self, "_optimization_eligible", bool(self._trajectory_eligible and self.role == "learner"
            and self.purpose in ("synthetic", "training") and self.boundary in ("natural", "cutoff")))

    @property
    def trajectory_eligible(self):
        """Contiguous targets from this history, independent of optimizer admission."""
        return self._trajectory_eligible

    @property
    def optimization_eligible(self):
        return self._optimization_eligible

    def to_dict(self):
        result = {"episode": self.episode, "player": self.player, "role": self.role,
            "history": [c.encode_frame(frame) for frame in self.history], "decisions": [item.to_dict() for item in self.decisions],
            "samples": [item.to_dict(schema=self.trajectory_schema) for item in self.samples], "boundary": self.boundary,
            "history_sha256": self.history_sha256, "evidence_ref": self.evidence_ref,
            "behavior_checkpoint": self.behavior_checkpoint, "opponent_checkpoint": self.opponent_checkpoint,
            "weights_sha256": self.weights_sha256, "opponent_weights_sha256": self.opponent_weights_sha256,
            "opponent_pool": self.opponent_pool, "purpose": self.purpose, "origin": self.origin,
            "estimates": [item.to_dict() for item in self.estimates]}
        if self.trajectory_schema != c.TRAJECTORY_SCHEMA: result["trajectory_schema"] = self.trajectory_schema
        return result

    @classmethod
    def from_dict(cls, raw, *, trajectory_schema=c.TRAJECTORY_SCHEMA):
        if trajectory_schema not in _TRAJECTORIES.values(): raise ValueError("Unsupported trajectory schema")
        names = {item.name for item in cls.__dataclass_fields__.values() if item.init}
        if trajectory_schema == c.TRAJECTORY_SCHEMA: names.remove("trajectory_schema")
        data = _record(raw, names)
        if data.get("trajectory_schema", c.TRAJECTORY_SCHEMA) != trajectory_schema:
            raise ValueError("Rollout trajectory disagrees with its artifact schema")
        if any(type(data[name]) is not list for name in ("history", "decisions", "samples", "estimates")):
            raise ValueError("Rollout sequences must be JSON arrays")
        history = tuple(c.decode_frame(frame) for frame in data["history"])
        decisions = tuple(BehaviorDecision.from_dict(record, history[record["decision_seq"]]) for record in data["decisions"])
        by_seq = {record.decision_seq: record for record in decisions}
        samples = []
        for raw_sample in data["samples"]:
            keys = ("decision_seq", "result_seq", "successor_seq", "reward", "bootstrap_value", "eligible")
            if trajectory_schema == c.TERMINAL_TRAJECTORY_SCHEMA: keys += ("terminal_closure",)
            sample = _record(raw_sample, keys)
            if sample.get("terminal_closure") is not None:
                closure = _record(sample["terminal_closure"], ("consumed_seq", "superseded_seq", "terminal_seq", "finalized_seq"))
                sample["terminal_closure"] = c.TerminalClosure(**closure)
            sample["decision"] = by_seq[sample.pop("decision_seq")]
            samples.append(RolloutSample(**sample))
        estimates = []
        for raw_estimate in data["estimates"]:
            estimate = _record(raw_estimate, ("decision_seq", "value", "prefix"))
            estimates.append(DecisionEstimate(estimate["decision_seq"], estimate["value"], RecurrentPrefix(**estimate["prefix"])))
        return cls(**{**data, "history": history, "decisions": decisions, "samples": tuple(samples), "estimates": tuple(estimates)})


class ContextualSession:
    """One player's incremental history and private recurrent state."""

    def __init__(self, model, episode, player, checkpoint, opponent_checkpoint, *, schema=c.TOWN_SCHEMA, seed=0):
        _text(checkpoint, "checkpoint"); _text(opponent_checkpoint, "opponent checkpoint")
        self.model, self.checkpoint, self.opponent_checkpoint = model, checkpoint, opponent_checkpoint
        self.weights_sha256 = weights_sha256(model)
        self.encoder = ContextualEncoder(episode, player, schema)
        self.history = c.PlayerHistory(episode, player, schema=schema)
        self._state = model.initial_state().detach().clone()
        self._digest = hashlib.sha256()
        self._values, self._prefixes, self._policies, self._choices = {}, {}, {}, {}
        c._integer(seed, high=2**63 - 1)
        self.generator = torch.Generator(device=self._state.device).manual_seed(seed)
        self.finished = False
        self._broken = False

    @property
    def state(self):
        return self._state.detach().clone()

    def _check_weights(self):
        if self._broken: raise ValueError("Session failed; partial collection cannot be resumed")
        if weights_sha256(self.model) != self.weights_sha256:
            self._broken = True
            raise ValueError("Behavior weights changed during collection")

    def consume(self, frames):
        if self.finished: raise ValueError("Session is finished")
        self._check_weights()
        try:
            self._consume(frames)
        except Exception:
            self._broken = True
            raise
        self._check_weights()

    def _consume(self, frames):
        for raw in frames:
            frame = raw if type(raw) is c.Frame else c.decode_frame(raw)
            if frame.seq < len(self.history.frames):
                if self.history.frames[frame.seq] != frame: raise ValueError("Previously delivered history prefix changed")
                continue
            # Contract/encoder failures abort collection; a failed episode is
            # never recoverable as an eligible partial rollout.
            self.history.append(frame)
            encoded = self.encoder.encode(frame)
            with torch.no_grad(): output = self.model.forward_step(encoded, self._state)
            self._state = output.state.detach().clone()
            self._digest.update(_frame_bytes(frame))
            if frame.kind == "decision":
                if output.decision is None: raise ValueError("Model omitted decision policy/value")
                value = float(output.decision.value.detach().item()); _finite(value, "behavior value")
                if output.decision.distribution.support != _support(frame): raise ValueError("Model changed contextual support")
                self._values[frame.seq] = value
                self._prefixes[frame.seq] = RecurrentPrefix(frame.episode, frame.player, self.checkpoint,
                    self.weights_sha256, frame.seq, self._digest.hexdigest())
                self._policies = {frame.seq: output.decision.distribution}
            elif output.decision is not None: raise ValueError("Only decisions may produce policy/value output")

    def choose(self, decision_seq):
        self._check_weights()
        if self.finished or self.history.decision is None or self.history.decision.seq != decision_seq or self.history.action is not None:
            raise ValueError("Only the current unacted own decision may be chosen")
        if decision_seq in self._choices: return self._choices[decision_seq]
        distribution = self._policies[decision_seq]
        with torch.no_grad():
            action = distribution.sample(generator=self.generator)
            evaluated = distribution.evaluate(action)
        record = BehaviorDecision(decision_seq, action, distribution.support, float(evaluated.log_prob.detach().item()),
            self._values[decision_seq], tuple(ConditionalRecord(item.name, item.selected_index, item.support_size,
                float(item.log_prob.detach().item())) for item in evaluated.conditionals),
            self._prefixes[decision_seq], self.opponent_checkpoint)
        self._choices[decision_seq] = record
        self._check_weights()
        return record


def _paired_binding(histories, manifest):
    if type(manifest) is not dict or len(histories) != 2:
        raise ValueError("Collection requires two histories and a final manifest")
    red, blue = histories
    if (red.player, blue.player) != (0, 1) or red.episode != blue.episode or red.schema != blue.schema or red.boundary != blue.boundary:
        raise ValueError("Paired histories disagree on identity/schema/boundary")
    profile = red.profile
    expected = {"schema_version": 1, "kind": "contextual-headless-self-play", "environment": profile.environment,
        "history_schema": profile.history_schema, "action_schema": profile.action_schema, "episode": red.episode}
    if any(type(manifest.get(key)) is not type(value) or manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Collection manifest identity/schema mismatch")
    status = {"natural": "completed", "cutoff": "truncated", "failure": "failed"}.get(red.boundary)
    if status is None or manifest.get("status") != status: raise ValueError("Unfinalized or mismatched collection boundary")
    _text(manifest.get("evidence_ref"), "finalization evidence")
    if red.boundary in ("natural", "cutoff"):
        name = "verification" if red.boundary == "natural" else "cutoff_verification"
        proofs = tuple(getattr(h.frames[-1], name) for h in histories)
        prefixes = tuple(hashlib.sha256(b"".join(_frame_bytes(f) for f in h.frames[:-1])).hexdigest() for h in histories)
        if (proofs[0] is None or proofs[0] != proofs[1] or proofs[0].history_sha256 != prefixes
                or proofs[0].evidence_ref != manifest["evidence_ref"]):
            raise ValueError("Both final proofs must bind the complete player prefixes")
    expected_outcomes = {color: h.terminal if status == "completed" else None for color, h in zip(COLORS, histories)}
    expected_rewards = {color: 1 if value == "win" else -1 if value == "loss" else None for color, value in expected_outcomes.items()}
    if manifest.get("outcomes_by_player") != expected_outcomes or manifest.get("rewards_by_player") != expected_rewards:
        raise ValueError("Manifest rewards/outcomes disagree with verified boundaries")
    if status == "completed" and (type(manifest.get("returncode")) is not int or manifest["returncode"] != 0):
        raise ValueError("Natural collection requires clean process exit")
    for color, h in zip(COLORS, histories):
        declaration = manifest.get("players", {}).get(color, {}).get(profile.history_key, {})
        if declaration.get("canonical_sha256") != h.digest() or declaration.get("boundary") != h.boundary:
            raise ValueError("Manifest does not bind the full player history")
        prefix = h.frames[:-1] if h.frames[-1].kind in ("finalized", "cutoff_finalized") else h.frames
        digest = hashlib.sha256(b"".join(_frame_bytes(frame) for frame in prefix)).hexdigest()
        if declaration.get("prefix_sha256") != digest:
            raise ValueError("Manifest does not bind the exact pre-finalization prefix")
        if status != "failed" and declaration.get("complete") is not True:
            raise ValueError("Eligible collection requires complete player histories")


@dataclass(frozen=True)
class CollectedEpisode:
    rollouts: tuple[ContextualRollout, ContextualRollout]
    manifest_json: str
    audit_json: str
    model_configs_json: tuple[str, str]
    artifact_schema: str = ROLLOUT_SCHEMA

    def __post_init__(self):
        _tuple(self.rollouts, ContextualRollout)
        if len(self.rollouts) != 2: raise ValueError("Collected episode requires both seats")
        if self.artifact_schema not in _TRAJECTORIES or any(
                r.trajectory_schema != _TRAJECTORIES[self.artifact_schema] for r in self.rollouts):
            raise ValueError("Collected artifact/trajectory versions disagree")
        red, blue = self.rollouts
        if ({red.role, blue.role} != {"learner", "frozen_opponent"}
                or (red.opponent_pool, red.purpose, red.origin) != (blue.opponent_pool, blue.purpose, blue.origin)
                or red.opponent_checkpoint != blue.behavior_checkpoint or blue.opponent_checkpoint != red.behavior_checkpoint
                or red.opponent_weights_sha256 != blue.weights_sha256 or blue.opponent_weights_sha256 != red.weights_sha256):
            raise ValueError("Paired frozen-opponent identity/role/purpose mismatch")
        histories = []
        for rollout in self.rollouts:
            history = c.PlayerHistory(rollout.episode, rollout.player, schema=rollout.history[0].schema)
            for frame in rollout.history: history.append(frame)
            histories.append(history)
        _paired_binding(histories, json.loads(self.manifest_json))
        _tuple(self.model_configs_json, str)
        if len(self.model_configs_json) != 2: raise ValueError("Both model configurations are required")
        for value, history in zip(self.model_configs_json, histories):
            if ModelConfig.from_dict(json.loads(value)).schema != model_schema_for(history.schema):
                raise ValueError("Recorded model version disagrees with the player history profile")
        json.loads(self.audit_json)

    def to_dict(self):
        return {"schema": self.artifact_schema, "rollouts": [rollout.to_dict() for rollout in self.rollouts],
            "manifest": json.loads(self.manifest_json), "audit": json.loads(self.audit_json),
            "model_configs": [json.loads(value) for value in self.model_configs_json]}

    @classmethod
    def from_dict(cls, raw):
        data = _record(raw, ("schema", "rollouts", "manifest", "audit", "model_configs"))
        if data["schema"] not in _TRAJECTORIES or type(data["rollouts"]) is not list or type(data["model_configs"]) is not list:
            raise ValueError("Unsupported collection artifact schema")
        return cls(tuple(ContextualRollout.from_dict(item, trajectory_schema=_TRAJECTORIES[data["schema"]]) for item in data["rollouts"]),
                   _canonical(data["manifest"]), _canonical(data["audit"]),
                   tuple(_canonical(config) for config in data["model_configs"]), data["schema"])


class ContextualCollector:
    """Fixed two-policy collection; run_episode never invokes an optimizer."""

    def __init__(self, models, checkpoints, *, learner_player, opponent_pool, interface=c.TOWN_ENVIRONMENT,
                 purpose="diagnostic", origin="native", policy_seeds=(0, 1), trajectory_schema=c.TERMINAL_TRAJECTORY_SCHEMA):
        if type(models) is not tuple or len(models) != 2 or type(checkpoints) is not tuple or len(checkpoints) != 2:
            raise ValueError("Collection requires fixed models/checkpoints for both seats")
        c._integer(learner_player, high=1); _text(opponent_pool, "opponent pool")
        if purpose not in ("diagnostic", "synthetic", "training") or origin not in ("native", "synthetic"):
            raise ValueError("Explicit collection purpose/origin is required")
        if purpose == "synthetic" and origin != "synthetic": raise ValueError("Synthetic data cannot identify a native run")
        if type(policy_seeds) is not tuple or len(policy_seeds) != 2: raise ValueError("Both policy seeds are required")
        self.profile = c.profile_for(interface)
        if interface != self.profile.environment: raise ValueError("Collector requires an environment interface")
        if any(model.config.schema != model_schema_for(interface) for model in models):
            raise ValueError("Both frozen collection models must match the explicit interface")
        self.models, self.checkpoints = models, checkpoints
        if trajectory_schema not in _TRAJECTORIES.values(): raise ValueError("Unsupported collector trajectory schema")
        self.trajectory_schema = trajectory_schema
        self.learner_player, self.opponent_pool, self.purpose, self.origin = learner_player, opponent_pool, purpose, origin
        self.policy_seeds = policy_seeds
        self.sessions = None
        self._episodes = set()

    def start(self, episode):
        _text(episode, "episode")
        if episode in self._episodes or self.sessions is not None and not all(s.finished for s in self.sessions):
            raise ValueError("Finish prior collection and use a fresh episode identity")
        self.sessions = tuple(ContextualSession(model, episode, player, self.checkpoints[player], self.checkpoints[1 - player],
            schema=self.profile.history_schema, seed=self.policy_seeds[player]) for player, model in enumerate(self.models))
        self._episodes.add(episode)

    def act(self, request):
        if (self.sessions is None or request.done or request.kind != "contextual" or request.past_horizon
                or type(request.player) is not int or request.player not in (0, 1) or request.color != COLORS[request.player]
                or request.reward != 0 or not request.frames):
            raise ValueError("Expected a current own contextual request before the horizon")
        session = self.sessions[request.player]
        session.consume(request.frames)
        last = c.decode_frame(request.frames[-1])
        if last.kind != "decision" or c.decode_frame(request.record) != last:
            raise ValueError("Request must end at its exact decision frame")
        return session.choose(last.seq).action

    def finish(self, end, *, histories, manifest_path=None):
        if self.sessions is None or not end.done or type(histories) is not tuple or len(histories) != 2:
            raise ValueError("Finish requires both complete contextual histories")
        for session, frames in zip(self.sessions, histories): session.consume(frames)
        checked = tuple(session.history for session in self.sessions)
        _paired_binding(checked, end.manifest)
        if end.status != end.manifest["status"] or end.outcomes != end.manifest["outcomes_by_player"] or end.rewards != end.manifest["rewards_by_player"]:
            raise ValueError("EpisodeEnd disagrees with its finalized manifest")
        for color, session in zip(COLORS, self.sessions):
            bootstrap = end.bootstrap.get(color)
            if session.history.boundary == "cutoff":
                cutoff = next(frame for frame in session.history.frames if frame.kind == "cutoff")
                if bootstrap is None or c.decode_frame(bootstrap) != session.history.frames[cutoff.decision_seq]:
                    raise ValueError("Cutoff bootstrap is not the actual own decision")
            elif bootstrap is not None: raise ValueError("Only verified cutoffs may bootstrap")
        audit_report = {"source": "synthetic", "native_evidence_verified": False}
        if self.origin == "native":
            if manifest_path is None: raise ValueError("Native collection requires independent persisted-manifest audit")
            path = Path(manifest_path)
            persisted = json.loads(path.read_text())
            if persisted != end.manifest: raise ValueError("Persisted native manifest changed")
            from validation.probes.audit_contextual_histories import audit
            paths = tuple(path.parent / persisted["players"][color][self.profile.history_key]["path"] for color in COLORS)
            audit_report = audit(*paths, path)
            if not audit_report["consistent"] or not audit_report["manifest_bound"]:
                raise ValueError("Independent native history audit failed")
        rollouts = []
        for player, session in enumerate(self.sessions):
            samples = []
            for transition in session.history.transitions(schema=self.trajectory_schema):
                record = session._choices.get(transition.decision_seq)
                if record is None: raise ValueError("Resolved action has no sampled behavior record")
                value = session._values.get(transition.successor_seq) if transition.bootstrap_allowed else None
                samples.append(RolloutSample(record, transition.result_seq, transition.successor_seq, transition.reward, value,
                                             transition.eligible, transition.terminal_closure))
            rollouts.append(ContextualRollout(session.history.episode, player,
                "learner" if player == self.learner_player else "frozen_opponent", session.history.frames,
                tuple(session._choices.values()), tuple(samples), session.history.boundary, session.history.digest(),
                end.manifest["evidence_ref"], session.checkpoint, session.opponent_checkpoint, session.weights_sha256,
                self.sessions[1 - player].weights_sha256, self.opponent_pool, self.purpose, self.origin,
                tuple(DecisionEstimate(seq, value, session._prefixes[seq]) for seq, value in session._values.items()),
                self.trajectory_schema))
        result = CollectedEpisode(tuple(rollouts), _canonical(end.manifest), _canonical(audit_report),
            tuple(_canonical(model.config.to_dict()) for model in self.models),
            next(schema for schema, trajectory in _TRAJECTORIES.items() if trajectory == self.trajectory_schema))
        for session in self.sessions: session._check_weights(); session.finished = True
        return result

    def run_episode(self, env, *, episode, seed, max_requests=None):
        if getattr(env, "interface", None) != self.profile.environment:
            raise ValueError("Collector/environment interface mismatch")
        if max_requests is not None: c._integer(max_requests, low=1)
        self.start(str(episode))
        request = env.reset(episode=episode, seed=seed)
        choices = 0
        while not request.done:
            if max_requests is not None and choices >= max_requests:
                request = env.truncate("collection_window")
            else:
                request = env.step(self.act(request)); choices += 1
        return self.finish(request, histories=(env.history("red"), env.history("blue")),
                           manifest_path=env.episode_dir / "manifest.json" if self.origin == "native" else None)
