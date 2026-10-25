"""Per-player data contract, not a collector, optimizer or engine verifier.

Only the environment's trusted finalizer can attest a natural result. The
checks here detect inconsistent assertions; Python records do not establish
that engine events happened or that the producer preserved player visibility.
Recurrent provenance is bookkeeping, never an input to the actor or critic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

from .actions import CandidateSet, movement_candidates
from .contract import Frame, HistoryCursor, integer


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Missing {name}")


def _finite(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _digest(value: str) -> None:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected a lowercase SHA-256 history digest")


def _history(history: tuple[Frame, ...]) -> tuple[HistoryCursor, dict[int, str]]:
    if type(history) is not tuple or not history or type(history[0]) is not Frame:
        raise ValueError("A trajectory needs an immutable, nonempty typed history")
    cursor = HistoryCursor(history[0].episode, history[0].player)
    digest = hashlib.sha256()
    prefixes = {}
    for frame in history:
        cursor.consume(frame)
        digest.update(json.dumps(asdict(frame), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        prefixes[frame.seq] = digest.hexdigest()
    return cursor, prefixes


@dataclass(frozen=True)
class RecurrentProvenance:
    """History consumed from zero through this frame, under one checkpoint.

    A consumer must replay this prefix (or separately verify its stored state).
    This reference alone cannot prove which recurrent tensor was actually used.
    """

    episode: str
    player: int
    checkpoint: str
    through_seq: int
    history_sha256: str

    def __post_init__(self):
        _text(self.episode, "episode identity")
        integer(self.player, 0, 1, "player")
        _text(self.checkpoint, "checkpoint identity")
        integer(self.through_seq, 0, 2**63 - 1, "history prefix sequence")
        _digest(self.history_sha256)


def recurrent_provenance(history: tuple[Frame, ...], checkpoint: str) -> RecurrentProvenance:
    """Reference the exact complete player prefix, including its latest frame."""
    cursor, prefixes = _history(history)
    return RecurrentProvenance(cursor.episode, cursor.player, checkpoint,
                               history[-1].seq, prefixes[history[-1].seq])


@dataclass(frozen=True)
class DecisionRecord:
    """Movement-v1 behavior sample with its exact versioned candidate support.

    Generic candidate scoring does not yet imply a generic action collector:
    the bridge history and selected action still use movement-v1 routing.
    """

    frame: Frame
    action: int
    behavior_log_prob: float
    behavior_value: float
    behavior_checkpoint: str
    opponent_checkpoint: str
    recurrent_prefix: RecurrentProvenance
    action_candidates: CandidateSet | None = None

    def __post_init__(self):
        if type(self.frame) is not Frame or self.frame.kind != "decision":
            raise ValueError("DecisionRecord requires a typed decision frame with its mask")
        integer(self.action, 0, 8, "sampled action")
        if not self.frame.mask[self.action]:
            raise ValueError("Sampled action is absent from the behavior attempt mask")
        expected_candidates = movement_candidates(self.frame.mask)
        if self.action_candidates is None:
            object.__setattr__(self, "action_candidates", expected_candidates)
        elif (type(self.action_candidates) is not CandidateSet
              or self.action_candidates != expected_candidates):
            raise ValueError("Behavior candidates must match exact movement-v1 support and parameters")
        _finite(self.behavior_log_prob, "Behavior log-probability")
        if self.behavior_log_prob > 0:
            raise ValueError("Behavior log-probability cannot be positive")
        _finite(self.behavior_value, "Behavior value")
        _text(self.behavior_checkpoint, "behavior checkpoint")
        _text(self.opponent_checkpoint, "opponent checkpoint")
        if type(self.recurrent_prefix) is not RecurrentProvenance:
            raise ValueError("Decision requires explicit recurrent prefix provenance")
        if (self.recurrent_prefix.episode, self.recurrent_prefix.player,
                self.recurrent_prefix.through_seq, self.recurrent_prefix.checkpoint) != (
                self.frame.episode, self.frame.player, self.frame.seq, self.behavior_checkpoint):
            raise ValueError("Decision recurrent provenance does not match its player/checkpoint/frame")

    @property
    def selected_candidate_key(self) -> str:
        return str(self.action)


@dataclass(frozen=True)
class TerminalVerification:
    """Assertions from the trusted finalizer, indexed by player 0 then player 1.

    ``evidence_ref`` identifies retained engine/socket/process evidence. Complete
    player histories must be verified independently of optional video success.
    Do not construct this from a socket terminal or a missing hero alone.
    """

    episode: str
    evidence_ref: str
    engine_outcomes: tuple[Literal["win", "loss"], Literal["win", "loss"]]
    socket_outcomes: tuple[Literal["win", "loss"], Literal["win", "loss"]]
    process_exit_code: int
    histories_complete: tuple[bool, bool]
    history_last_seq: tuple[int, int]
    history_sha256: tuple[str, str]

    def __post_init__(self):
        _text(self.episode, "episode identity")
        _text(self.evidence_ref, "trusted finalization evidence reference")
        if (type(self.engine_outcomes) is not tuple
                or self.engine_outcomes not in (("win", "loss"), ("loss", "win"))
                or type(self.socket_outcomes) is not tuple
                or self.socket_outcomes != self.engine_outcomes):
            raise ValueError("Both engine and socket outcomes must agree and be complementary")
        if type(self.process_exit_code) is not int or self.process_exit_code != 0:
            raise ValueError("Natural finalization requires a clean engine process exit")
        if (type(self.histories_complete) is not tuple or len(self.histories_complete) != 2
                or any(v is not True for v in self.histories_complete)):
            raise ValueError("Both player histories must be complete independently of video")
        if type(self.history_last_seq) is not tuple or len(self.history_last_seq) != 2:
            raise ValueError("Finalization requires both final player history sequences")
        for seq in self.history_last_seq:
            integer(seq, 0, 2**63 - 1, "final history sequence")
        if type(self.history_sha256) is not tuple or len(self.history_sha256) != 2:
            raise ValueError("Finalization requires both player history hashes")
        for digest in self.history_sha256:
            _digest(digest)


@dataclass(frozen=True)
class NaturalEnd:
    verification: TerminalVerification

    def __post_init__(self):
        if type(self.verification) is not TerminalVerification:
            raise ValueError("Natural ending requires trusted finalization evidence")


@dataclass(frozen=True)
class DecisionCutoff:
    """An intentional rollout horizon at a real, unacted next decision.

    This cannot represent a timeout without such a decision or a fabricated
    final observation. The prefix includes every visible intervening event.
    """

    frame: Frame
    bootstrap_value: float
    recurrent_prefix: RecurrentProvenance

    def __post_init__(self):
        if type(self.frame) is not Frame or self.frame.kind != "decision":
            raise ValueError("Cutoff requires a real next decision frame")
        _finite(self.bootstrap_value, "Cutoff bootstrap value")
        if type(self.recurrent_prefix) is not RecurrentProvenance:
            raise ValueError("Cutoff requires recurrent prefix provenance")


@dataclass(frozen=True)
class TechnicalFailure:
    """Unscored failure/timeout; even earlier decisions in this item are excluded."""

    reason: str

    def __post_init__(self):
        _text(self.reason, "technical failure reason")


@dataclass(frozen=True)
class TransitionTarget:
    decision_seq: int
    next_decision_seq: int | None
    reward: float | None
    bootstrap_value: float | None
    return_target: float | None
    eligible: bool


@dataclass(frozen=True)
class PlayerTrajectory:
    """One seat with full history from episode start, including visible events.

    No opponent actions are interleaved. Frozen opponent trajectories may be
    retained for evaluation, but they never receive on-policy return targets.
    Checkpoint changes require a new collection; an episode cannot switch them.
    """

    episode: str
    player: int
    behavior_checkpoint: str
    opponent_checkpoint: str
    role: Literal["learner", "frozen_opponent"]
    history: tuple[Frame, ...]
    decisions: tuple[DecisionRecord, ...]
    boundary: NaturalEnd | DecisionCutoff | TechnicalFailure

    def __post_init__(self):
        _text(self.episode, "episode identity")
        integer(self.player, 0, 1, "player")
        _text(self.behavior_checkpoint, "behavior checkpoint")
        _text(self.opponent_checkpoint, "opponent checkpoint")
        if self.role not in ("learner", "frozen_opponent"):
            raise ValueError("Unknown collection role")
        cursor, prefixes = _history(self.history)
        if (cursor.episode, cursor.player) != (self.episode, self.player):
            raise ValueError("Trajectory history belongs to a different episode/player")
        if (type(self.decisions) is not tuple or not self.decisions
                or any(type(d) is not DecisionRecord for d in self.decisions)):
            raise ValueError("Trajectory requires an immutable, nonempty tuple of decision records")
        taken = [f for f in self.history if f.kind == "action"]
        if len(taken) != len(self.decisions):
            raise ValueError("Every taken decision must have exactly one behavior record")
        for record, action in zip(self.decisions, taken):
            frame = record.frame
            if (record.behavior_checkpoint, record.opponent_checkpoint) != (
                    self.behavior_checkpoint, self.opponent_checkpoint):
                raise ValueError("Behavior and frozen opponent checkpoint identities must stay fixed")
            if (frame.seq != action.decision_seq or record.action != action.action
                    or frame != self.history[action.decision_seq]):
                raise ValueError("Behavior records must match taken decisions in player-history order")
            self._match_prefix(record.recurrent_prefix, frame, prefixes)

        if type(self.boundary) is NaturalEnd:
            final = self.boundary.verification
            if (final.episode != self.episode
                    or final.history_last_seq[self.player] != self.history[-1].seq
                    or final.history_sha256[self.player] != prefixes[self.history[-1].seq]):
                raise ValueError("Natural finalization must reference this complete player history")
            # A terminal may arrive before its last action_result. Its action
            # remains the final transition; no invented result/view is required.
            # It may also supersede an emitted decision before an action was
            # taken. Keep that observed frame in history, but create no sample:
            # the terminal reward belongs to the last actually taken action.
        elif type(self.boundary) is DecisionCutoff:
            if self.boundary.frame != self.history[-1] or cursor.decision != self.boundary.frame:
                raise ValueError("Cutoff must be the next unacted decision of this player's history")
            self._match_prefix(self.boundary.recurrent_prefix, self.boundary.frame, prefixes)
        elif type(self.boundary) is not TechnicalFailure:
            raise ValueError("Missing explicit natural, next-decision or technical boundary")

    def _match_prefix(self, prefix: RecurrentProvenance, frame: Frame, prefixes: dict[int, str]) -> None:
        if (prefix.episode, prefix.player, prefix.checkpoint, prefix.through_seq,
                prefix.history_sha256) != (
                self.episode, self.player, self.behavior_checkpoint, frame.seq, prefixes[frame.seq]):
            raise ValueError("Recurrent provenance must match the exact same-player history/checkpoint")

    @property
    def eligible(self) -> bool:
        return self.role == "learner" and type(self.boundary) is not TechnicalFailure

    def transition_targets(self) -> tuple[TransitionTarget, ...]:
        """Reference undiscounted (gamma=1, lambda=1) returns, with no shaping.

        This is target semantics for tests, not an implementation of PPO/GAE.
        Ordinary transitions end at this player's next decision, potentially
        after a whole opponent turn. A true terminal is the only zero bootstrap.
        """
        if type(self.boundary) is TechnicalFailure:
            return tuple(TransitionTarget(d.frame.seq, None, None, None, None, False)
                         for d in self.decisions)
        terminal = type(self.boundary) is NaturalEnd
        final_reward = (1.0 if self.boundary.verification.engine_outcomes[self.player] == "win"
                        else -1.0) if terminal else 0.0
        final_value = 0.0 if terminal else self.boundary.bootstrap_value
        total_return = final_reward + final_value
        targets = []
        for index, decision in enumerate(self.decisions):
            last = index == len(self.decisions) - 1
            following = None if last else self.decisions[index + 1]
            next_seq = (None if terminal else self.boundary.frame.seq) if last else following.frame.seq
            value = final_value if last else following.behavior_value
            targets.append(TransitionTarget(
                decision.frame.seq, next_seq, final_reward if last else 0.0,
                value if self.eligible else None, total_return if self.eligible else None, self.eligible))
        return tuple(targets)
