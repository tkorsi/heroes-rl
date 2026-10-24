"""Strict contextual history contract; no engine adapter or learning updates.

All observations must already have been filtered by an audited player callback.
Validation checks shape and chronology, not whether a producer told the truth
about visibility. Routing handles, revisions and evidence are bookkeeping, never
numeric neural features. Frozen records preserve every delivered prefix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re

SCHEMA = "heroes-player-history-v2"
ACTION_SCHEMA = "heroes-context-actions-v2"
ENVIRONMENT = "adventure-windows-v1"
MAX_INT = 2**63 - 1


def _integer(value, low=0, high=MAX_INT, name="integer"):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")


def _text(value, name="handle"):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"Invalid {name}")


def _tuple(value, member_type, name):
    if type(value) is not tuple or any(type(x) is not member_type for x in value):
        raise ValueError(f"{name} must be an immutable tuple of {member_type.__name__}")


def _position(value):
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("Position must be an immutable (x, y, z)")
    for x in value:
        _integer(x)


def _unique(values, name):
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {name}")


@dataclass(frozen=True)
class Context:
    kind: str = "map"
    subject: str | None = None

    def __post_init__(self):
        if self.kind not in ("map", "hero", "town"):
            raise ValueError("Unsupported context")
        if self.kind == "map":
            if self.subject is not None:
                raise ValueError("Map context has no subject")
        else:
            _text(self.subject)


@dataclass(frozen=True)
class Action:
    kind: str
    subject: str | None = None
    x: int | None = None
    y: int | None = None
    schema: str = ACTION_SCHEMA

    def __post_init__(self):
        if self.schema != ACTION_SCHEMA or self.kind not in (
                "move_to", "open_hero", "open_town", "close", "end_turn"):
            raise ValueError("Unsupported contextual action")
        if self.kind in ("move_to", "open_hero", "open_town"):
            _text(self.subject)
        elif self.subject is not None:
            raise ValueError("This action has no subject")
        if self.kind == "move_to":
            _integer(self.x)
            _integer(self.y)
        elif self.x is not None or self.y is not None:
            raise ValueError("Only destination movement takes coordinates")


@dataclass(frozen=True)
class Tile:
    position: tuple[int, int, int]
    terrain: int
    road: int
    blocked: bool
    visitable: bool

    def __post_init__(self):
        _position(self.position)
        _integer(self.terrain, high=255)
        _integer(self.road, high=255)
        if type(self.blocked) is not bool or type(self.visitable) is not bool:
            raise ValueError("Tile flags must be boolean")


@dataclass(frozen=True)
class ObjectView:
    handle: str
    position: tuple[int, int, int]
    kind: int
    subtype: int
    owner: str

    def __post_init__(self):
        _text(self.handle)
        _position(self.position)
        _integer(self.kind, high=2**31 - 1)
        _integer(self.subtype, high=2**31 - 1)
        if self.owner not in ("self", "opponent", "neutral"):
            raise ValueError("Object ownership must be relative")


@dataclass(frozen=True)
class Hero:
    handle: str
    position: tuple[int, int, int]
    movement: int
    movement_max: int | None = None

    def __post_init__(self):
        _text(self.handle)
        _position(self.position)
        _integer(self.movement)
        if self.movement_max is not None:
            _integer(self.movement_max)


@dataclass(frozen=True)
class Town:
    handle: str
    position: tuple[int, int, int]

    def __post_init__(self):
        _text(self.handle)
        _position(self.position)


@dataclass(frozen=True)
class Stack:
    slot: int
    creature: int
    count: int

    def __post_init__(self):
        _integer(self.slot, high=6)
        _integer(self.creature, high=2**31 - 1)
        _integer(self.count, low=1)


@dataclass(frozen=True)
class Window:
    context: Context
    army: tuple[Stack, ...]
    primary_skills: tuple[int, int, int, int] | None = None
    mana: int | None = None
    level: int | None = None
    buildings: tuple[int, ...] | None = None
    faction: int | None = None

    def __post_init__(self):
        if type(self.context) is not Context or self.context.kind == "map":
            raise ValueError("Window requires a hero or town context")
        _tuple(self.army, Stack, "army")
        _unique([s.slot for s in self.army], "army slot")
        if self.context.kind == "hero":
            if self.buildings is not None or self.faction is not None:
                raise ValueError("Hero window cannot expose town fields")
            if type(self.primary_skills) is not tuple or len(self.primary_skills) != 4:
                raise ValueError("Hero window requires four primary skills")
            for value in self.primary_skills:
                _integer(value)
            _integer(self.mana)
            _integer(self.level, low=1)
        else:
            if any(x is not None for x in (self.primary_skills, self.mana, self.level)):
                raise ValueError("Town window cannot expose hero fields")
            if self.buildings is not None:
                raise ValueError("Town building details are deferred in this interface")
            _integer(self.faction)


@dataclass(frozen=True)
class Observation:
    size: tuple[int, int, int]
    tiles: tuple[Tile, ...]
    objects: tuple[ObjectView, ...]
    resources: tuple[int, ...]
    heroes: tuple[Hero, ...]
    towns: tuple[Town, ...]
    window: Window | None = None

    def __post_init__(self):
        _position(self.size)
        if any(n <= 0 for n in self.size) or self.size[2] != 1:
            raise ValueError("First contextual interface requires one map level")
        for name, cls in (("tiles", Tile), ("objects", ObjectView), ("heroes", Hero), ("towns", Town)):
            _tuple(getattr(self, name), cls, name)
            for entry in getattr(self, name):
                if any(p >= n for p, n in zip(entry.position, self.size)):
                    raise ValueError("Observed position outside public map bounds")
        _unique([t.position for t in self.tiles], "visible tile")
        visible = {t.position for t in self.tiles}
        if any(obj.position not in visible for obj in self.objects):
            raise ValueError("Current object must be on a currently visible tile")
        for name in ("objects", "heroes", "towns"):
            _unique([entry.handle for entry in getattr(self, name)], name + " handle")
        _unique([x.handle for x in (*self.heroes, *self.towns)], "own subject handle")
        _tuple(self.resources, int, "resources")
        if len(self.resources) != 7:
            raise ValueError("Expected seven resource amounts")
        for amount in self.resources:
            _integer(amount)
        if self.window is not None:
            if type(self.window) is not Window:
                raise ValueError("Expected typed active window")
            own = self.heroes if self.window.context.kind == "hero" else self.towns
            if self.window.context.subject not in {entry.handle for entry in own}:
                raise ValueError("Window subject must be an accessible owned object")

    @property
    def context(self):
        return self.window.context if self.window is not None else Context()


@dataclass(frozen=True)
class Offers:
    """Public rectangular coordinate domain, never a reachable-target mask."""
    context: Context
    size: tuple[int, int, int]
    hero_refs: tuple[str, ...] = ()
    town_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.context) is not Context:
            raise ValueError("Offers require a typed context")
        _position(self.size)
        if any(n <= 0 for n in self.size) or self.size[2] != 1:
            raise ValueError("Invalid offer map bounds")
        for refs in (self.hero_refs, self.town_refs):
            _tuple(refs, str, "subject references")
            for ref in refs:
                _text(ref)
            _unique(refs, "offered subject")
        if self.context.kind != "map" and (self.hero_refs or self.town_refs):
            raise ValueError("First-stage windows offer close only")

    @classmethod
    def from_observation(cls, observation):
        if type(observation) is not Observation:
            raise ValueError("Expected typed observation")
        if observation.context.kind != "map":
            return cls(observation.context, observation.size)
        return cls(Context(), observation.size, tuple(h.handle for h in observation.heroes),
                   tuple(t.handle for t in observation.towns))

    def permits(self, action):
        if type(action) is not Action:
            return False
        if self.context.kind != "map":
            return action.kind == "close"
        if action.kind == "end_turn":
            return True
        if action.kind in ("open_hero", "move_to") and action.subject in self.hero_refs:
            return action.kind == "open_hero" or (action.x < self.size[0] and action.y < self.size[1])
        return action.kind == "open_town" and action.subject in self.town_refs


@dataclass(frozen=True)
class Result:
    status: str
    accepted: bool
    position: tuple[int, int, int] | None = None
    movement: int | None = None
    destination_reached: bool | None = None

    def __post_init__(self):
        if self.status not in ("reached", "movement_limit", "no_route_known", "interrupted",
                               "interaction", "hero_unavailable", "opened", "closed", "turn_ended", "rejected"):
            raise ValueError("Unsupported action result")
        if type(self.accepted) is not bool:
            raise ValueError("Result requires explicit action acceptance")
        if self.status == "rejected" and self.accepted:
            raise ValueError("Rejected result cannot be accepted")
        if self.status in ("opened", "closed", "turn_ended", "reached", "movement_limit", "no_route_known", "interaction") and not self.accepted:
            raise ValueError("Completed operation must be accepted")
        if self.position is not None:
            _position(self.position)
        if self.movement is not None:
            _integer(self.movement)
        if self.destination_reached is not None and type(self.destination_reached) is not bool:
            raise ValueError("destination_reached must be boolean")


NOTICE_TEXT_IDS = frozenset({
    "core.advevent.113", "hero.core.orrin.name", "hero.core.edric.name",
    "heroes.baseline.all_opponents_defeated", "heroes.baseline.opponent_won",
    "heroes.baseline.seven_days_without_town", "heroes.baseline.opponent_lost_last_town",
    *(f"core.restypes.{index}" for index in range(7)),
    *(f"core.mineevnt.{index}" for index in range(7)),
    *(f"core.genrltxt.{index}" for index in (1, 5, 7, 659)),
    *(f"core.arraytxt.{index}" for index in (*range(32, 42), *range(43, 58), *range(130, 136))),
})
# VCMI CCreature::getName{Singular,Plural}TextID uses the plural "creatures"
# namespace. The producer must still prove the delivered template/operation
# sequence and that the referenced creature is legitimate player information.
_CREATURE_NOTICE_TEXT_ID = re.compile(r"creatures\.core\.[A-Za-z][A-Za-z0-9_]{0,63}\.name\.(?:singular|plural)")
NOTICE_RESOURCES = ("wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold")


@dataclass(frozen=True)
class NoticeToken:
    """One delivered localized operation, never an unused MetaString pool entry."""
    op: str
    text_id: str | None = None
    number: int | None = None
    owner: str | None = None

    def __post_init__(self):
        if self.op in ("append_text", "replace_text"):
            if self.number is not None or (self.text_id is None) == (self.owner is None):
                raise ValueError("Notice text operation requires exactly one text ID or relative owner")
            if self.text_id is not None:
                if (type(self.text_id) is not str or self.text_id not in NOTICE_TEXT_IDS
                        and _CREATURE_NOTICE_TEXT_ID.fullmatch(self.text_id) is None):
                    raise ValueError("Unsupported notice text ID")
            elif self.owner not in ("self", "opponent"):
                raise ValueError("Notice player reference must be relative")
        elif self.op in ("append_number", "replace_number", "replace_positive_number"):
            if self.text_id is not None or self.owner is not None:
                raise ValueError("Notice number operation cannot carry text or owner")
            # Native REPLACE_POSITIVE_NUMBER formats signed values, including
            # zero and negative numbers; its name does not restrict the value.
            _integer(self.number, low=-(2**63), high=2**63 - 1, name="notice number")
        elif self.op == "newline":
            if any(value is not None for value in (self.text_id, self.number, self.owner)):
                raise ValueError("Notice newline has no payload")
        else:
            raise ValueError("Unsupported notice operation")


@dataclass(frozen=True)
class NoticeComponent:
    kind: str
    resource: str | None = None
    amount: int | None = None
    owner: str | None = None

    def __post_init__(self):
        if self.kind in ("resource", "resource_per_day"):
            if self.resource not in NOTICE_RESOURCES or self.owner is not None:
                raise ValueError("Resource notice component requires only a known resource and amount")
            _integer(self.amount, low=-(2**31), high=2**31 - 1, name="notice resource amount")
        elif self.kind == "flag":
            if self.resource is not None or self.amount is not None or self.owner not in ("self", "opponent"):
                raise ValueError("Flag notice component requires only a relative player")
        else:
            raise ValueError("Unsupported notice component")


@dataclass(frozen=True)
class Notice:
    """An informational engine notification: no reply, window operation or reward.

    ``modal`` is the native presentation mode, not authorization to fabricate a
    query acknowledgement. Ordered tokens describe only delivered operations;
    template semantics and entitlement remain the native producer's obligation.
    """
    display_mode: str
    text: tuple[NoticeToken, ...]
    components: tuple[NoticeComponent, ...]

    def __post_init__(self):
        if self.display_mode not in ("auto", "modal", "info"):
            raise ValueError("Unsupported informational notice display mode")
        _tuple(self.text, NoticeToken, "notice text")
        _tuple(self.components, NoticeComponent, "notice components")
        if len(self.text) > 64 or len(self.components) > 32:
            raise ValueError("Notice exceeds bounded token/component limits")


@dataclass(frozen=True)
class VisibleEvent:
    kind: str
    subject: str | None = None
    object_kind: int | None = None
    object_subtype: int | None = None
    owner: str | None = None
    side: str | None = None
    won: bool | None = None
    notice: Notice | None = None

    def __post_init__(self):
        if self.kind == "notice":
            if type(self.notice) is not Notice or any(value is not None for value in (
                    self.subject, self.object_kind, self.object_subtype, self.owner, self.side, self.won)):
                raise ValueError("Notice event requires only a typed notice")
            return
        if self.kind not in ("visit", "battle_start", "battle_complete", "turn_end", "hero_lost"):
            raise ValueError("Unsupported visible event")
        if self.notice is not None:
            raise ValueError("Only a notice event can carry notice content")
        if self.subject is not None:
            _text(self.subject)
        for value in (self.object_kind, self.object_subtype):
            if value is not None:
                _integer(value, high=2**31 - 1)
        if self.owner is not None and self.owner not in ("self", "opponent", "neutral"):
            raise ValueError("Event ownership must be relative")
        if self.side is not None and self.side not in ("attacker", "defender"):
            raise ValueError("Invalid battle side")
        if self.won is not None and type(self.won) is not bool:
            raise ValueError("Invalid battle result")


@dataclass(frozen=True)
class Verification:
    """Trusted finalizer attestation; audit metadata, not a model observation.

    Digests bind both complete prefixes BEFORE their finalized record. A terminal
    socket message alone cannot construct trustworthy evidence. The caller must
    independently verify process, engine and stream completion.
    """
    episode: str
    evidence_ref: str
    engine_outcomes: tuple[str, str]
    socket_outcomes: tuple[str, str]
    process_exit_code: int
    histories_complete: tuple[bool, bool]
    history_sha256: tuple[str, str]

    def __post_init__(self):
        _text(self.episode, "episode")
        _text(self.evidence_ref, "evidence reference")
        if (type(self.engine_outcomes) is not tuple or self.engine_outcomes not in
                (("win", "loss"), ("loss", "win")) or self.socket_outcomes != self.engine_outcomes
                or type(self.socket_outcomes) is not tuple):
            raise ValueError("Engine/socket results must agree and be complementary")
        if type(self.process_exit_code) is not int or self.process_exit_code != 0:
            raise ValueError("Natural finalization requires clean process exit")
        if type(self.histories_complete) is not tuple or self.histories_complete != (True, True) or any(
                type(v) is not bool for v in self.histories_complete):
            raise ValueError("Both player histories must be complete")
        _tuple(self.history_sha256, str, "history hashes")
        if len(self.history_sha256) != 2 or any(len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
                                              for d in self.history_sha256):
            raise ValueError("Expected two SHA-256 history hashes")


@dataclass(frozen=True)
class CutoffVerification:
    """Pair finalization of held decision prefixes after bounded continuation.

    Continuation is retained only in raw evidence, outside these learner prefixes.
    A requested process stop may be expected; shutdown_verified is the trusted
    finalizer's assertion that this was not a pre-existing crash or stream error.
    """
    episode: str
    evidence_ref: str
    histories_complete: tuple[bool, bool]
    failure_free: bool
    shutdown_verified: bool
    continuation_limit: int
    continuation_steps: tuple[int, int]
    history_sha256: tuple[str, str]

    def __post_init__(self):
        _text(self.episode, "episode")
        _text(self.evidence_ref, "evidence reference")
        if (type(self.histories_complete) is not tuple or self.histories_complete != (True, True)
                or any(type(v) is not bool for v in self.histories_complete)
                or self.failure_free is not True or self.shutdown_verified is not True):
            raise ValueError("Cutoff finalization requires complete, failure-free verified shutdown")
        _integer(self.continuation_limit)
        _tuple(self.continuation_steps, int, "continuation steps")
        if len(self.continuation_steps) != 2:
            raise ValueError("Continuation counts require both players")
        for steps in self.continuation_steps:
            _integer(steps, high=self.continuation_limit)
        _tuple(self.history_sha256, str, "history hashes")
        if len(self.history_sha256) != 2 or any(len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
                                              for d in self.history_sha256):
            raise ValueError("Expected two SHA-256 history hashes")


_PAYLOADS = {
    "view": {"observation"}, "event": {"event"}, "context": {"context", "revision", "decision_seq"},
    "context_invalidated": {"context", "revision", "reason"},
    "decision": {"observation", "offers", "revision"}, "submitted": {"action", "decision_seq"},
    "consumed": {"consumer", "decision_seq"}, "issued": {"command_index", "decision_seq"},
    "acknowledged": {"command_index", "accepted", "decision_seq"}, "result": {"result", "decision_seq"},
    "superseded": {"decision_seq", "reason"}, "terminal": {"outcome"}, "finalized": {"verification"},
    "cutoff": {"decision_seq", "reason"}, "cutoff_finalized": {"cutoff_verification"}, "failure": {"reason"},
}


@dataclass(frozen=True)
class Frame:
    episode: str
    player: int
    seq: int
    day: int
    kind: str
    observation: Observation | None = None
    context: Context | None = None
    offers: Offers | None = None
    revision: int | None = None
    decision_seq: int | None = None
    action: Action | None = None
    consumer: str | None = None
    result: Result | None = None
    event: VisibleEvent | None = None
    reason: str | None = None
    outcome: str | None = None
    verification: Verification | None = None
    cutoff_verification: CutoffVerification | None = None
    command_index: int | None = None
    accepted: bool | None = None
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or self.kind not in _PAYLOADS:
            raise ValueError("Unsupported contextual history schema/kind")
        _text(self.episode, "episode")
        _integer(self.player, high=1)
        _integer(self.seq)
        _integer(self.day, low=1)
        payload = {name for name in set().union(*_PAYLOADS.values()) if getattr(self, name) is not None}
        if payload != _PAYLOADS[self.kind]:
            raise ValueError("Missing or extraneous frame payload")
        for name, cls in (("observation", Observation), ("context", Context), ("offers", Offers),
                          ("action", Action), ("result", Result), ("event", VisibleEvent), ("verification", Verification),
                          ("cutoff_verification", CutoffVerification)):
            if getattr(self, name) is not None and type(getattr(self, name)) is not cls:
                raise ValueError(f"Expected typed {name}")
        for name in ("revision", "command_index"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name))
        if self.decision_seq is not None:
            _integer(self.decision_seq, high=self.seq - 1)
        if self.consumer is not None and self.consumer not in ("adapter", "engine"):
            raise ValueError("Consumption layer must be explicit")
        if self.outcome is not None and self.outcome not in ("win", "loss"):
            raise ValueError("Only natural win/loss terminal messages are supported")
        if self.reason is not None:
            _text(self.reason, "boundary reason")
        if self.accepted is not None and type(self.accepted) is not bool:
            raise ValueError("Native acknowledgement requires boolean acceptance")
        if self.kind == "decision" and self.offers != Offers.from_observation(self.observation):
            raise ValueError("Offers must preserve complete context and all coordinate support")


@dataclass(frozen=True)
class Transition:
    episode: str
    player: int
    decision_seq: int
    action: Action
    result_seq: int
    successor_seq: int | None
    reward: int | None
    eligible: bool
    bootstrap_allowed: bool
    boundary: str


class PlayerHistory:
    """Append-only one-player stream. It does not allocate or update neural state."""

    def __init__(self, episode: str, player: int):
        _text(episode, "episode")
        _integer(player, high=1)
        self.episode, self.player = episode, player
        self._frames = []
        self.context = Context()
        self.revision = -1
        self.day = 1
        self.decision = None
        self.action = None
        self.consumed = False
        self.commands = 0
        self.accepted_commands = 0
        self.native_rejected = False
        self.pending_command = None
        self.terminal = None
        self.boundary = None

    @property
    def frames(self):
        return tuple(self._frames)

    def digest(self):
        digest = hashlib.sha256()
        for frame in self._frames:
            digest.update(json.dumps(encode_frame(frame), sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def emit(self, kind, day=None, **payload):
        frame = Frame(self.episode, self.player, len(self._frames), self.day if day is None else day, kind, **payload)
        self.append(frame)
        return frame

    def append(self, frame):
        if type(frame) is not Frame or (frame.episode, frame.player) != (self.episode, self.player):
            raise ValueError("Cross-player/episode or untyped frame")
        if frame.seq != len(self._frames) or frame.day < self.day:
            raise ValueError("Missing, repeated, reordered or decreasing-day frame")
        kind = frame.kind
        if self.boundary is not None and not (self.boundary == "cutoff_pending" and kind in ("failure", "cutoff_finalized")):
            raise ValueError("History already finalized or cutoff prefix frozen")
        if self.terminal is not None and kind not in ("view", "event", "context_invalidated", "finalized", "failure"):
            raise ValueError("Only final visible observations or finalization may follow terminal")
        if kind in ("view", "decision"):
            if frame.observation.context != self.context:
                raise ValueError("Window fields need an explicit matching context change")
        if kind in ("view", "event") and self.decision is not None and self.action is None:
            raise ValueError("Supersede held decision before adding visible history")
        if kind == "decision":
            if self.decision is not None or self.action is not None:
                raise ValueError("Prior decision/action has not completed")
            if frame.revision <= self.revision:
                raise ValueError("Decision revisions must advance locally")
            self.revision = frame.revision
            self.decision = frame
        elif kind == "submitted":
            self._match_decision(frame)
            if self.action is not None or not self.decision.offers.permits(frame.action):
                raise ValueError("Action must match current contextual support")
            self.action = frame.action
        elif kind == "consumed":
            self._match_action(frame)
            if self.consumed:
                raise ValueError("Duplicate action consumption")
            self.consumed = True
        elif kind == "issued":
            self._match_consumed(frame)
            if self.action.kind not in ("move_to", "end_turn"):
                raise ValueError("Navigation cannot issue native game commands")
            if (self.pending_command is not None or frame.command_index != self.commands or self.native_rejected
                    or self.action.kind == "end_turn" and self.commands != 0):
                raise ValueError("Native commands require ordered acknowledgement; rejected steps stop execution")
            self.pending_command = frame.command_index
            self.commands += 1
        elif kind == "acknowledged":
            self._match_consumed(frame)
            if self.pending_command != frame.command_index:
                raise ValueError("Native acknowledgement has no matching issued command")
            self.pending_command = None
            self.accepted_commands += int(frame.accepted)
            self.native_rejected = self.native_rejected or not frame.accepted
        elif kind == "context":
            self._match_consumed(frame)
            expected = {"open_hero": Context("hero", self.action.subject),
                        "open_town": Context("town", self.action.subject)} if self.action.subject is not None else {}
            if self.action.kind == "close":
                expected["close"] = Context()
            if expected.get(self.action.kind) != frame.context or frame.revision <= self.revision:
                raise ValueError("Context change must match consumed navigation and advance revision")
            self.context, self.revision = frame.context, frame.revision
        elif kind == "context_invalidated":
            if (self.context.kind == "map" or frame.context != Context() or frame.revision <= self.revision
                    or self.decision is not None or self.action is not None
                    or frame.reason not in ("subject_unavailable", "terminal")):
                raise ValueError("Invalidation closes an unavailable window after explicit supersession")
            self.context, self.revision = frame.context, frame.revision
        elif kind == "result":
            self._match_consumed(frame)
            if self.pending_command is not None:
                raise ValueError("Action result precedes native acknowledgement")
            self._check_result(frame.result)
            self._clear_action()
        elif kind == "superseded":
            self._match_decision(frame)
            self._clear_action()
        elif kind == "terminal":
            if self.decision is not None or self.action is not None:
                raise ValueError("Explicitly supersede outstanding decisions/actions before terminal")
            self.terminal = frame.outcome
        elif kind == "finalized":
            proof = frame.verification
            if (self.terminal is None or proof.episode != self.episode
                    or proof.engine_outcomes[self.player] != self.terminal
                    or proof.history_sha256[self.player] != self.digest()):
                raise ValueError("Finalization does not bind this complete terminal history")
            self.boundary = "natural"
        elif kind == "cutoff":
            self._match_decision(frame)
            if self.action is not None:
                raise ValueError("Cutoff must be at this player's real unacted next decision")
            self.boundary = "cutoff_pending"
        elif kind == "cutoff_finalized":
            proof = frame.cutoff_verification
            if (self.boundary != "cutoff_pending" or proof.episode != self.episode
                    or proof.history_sha256[self.player] != self.digest()):
                raise ValueError("Cutoff finalization must bind this held-decision prefix")
            self.boundary = "cutoff"
        elif kind == "failure":
            self.boundary = "failure"
        self._frames.append(frame)
        self.day = frame.day

    def _match_decision(self, frame):
        if self.decision is None or frame.decision_seq != self.decision.seq:
            raise ValueError("Frame does not reference this player's current decision")

    def _match_action(self, frame):
        self._match_decision(frame)
        if self.action is None:
            raise ValueError("Action was not submitted")

    def _match_consumed(self, frame):
        self._match_action(frame)
        if not self.consumed:
            raise ValueError("Action was submitted but not consumed")

    def _clear_action(self):
        self.decision, self.action = None, None
        self.consumed, self.commands, self.accepted_commands, self.pending_command = False, 0, 0, None
        self.native_rejected = False

    def _check_result(self, result):
        action = self.action
        if action.kind == "move_to":
            if result.status not in ("reached", "movement_limit", "no_route_known", "interrupted",
                                     "interaction", "hero_unavailable", "rejected"):
                raise ValueError("Incorrect movement result")
            if result.status != "hero_unavailable" and (result.position is None or result.movement is None):
                raise ValueError("Movement results require observed position and points")
            if result.status == "reached" and result.destination_reached is not True:
                raise ValueError("Reached status requires observed arrival")
            if self.native_rejected and result.status not in ("interrupted", "rejected", "hero_unavailable"):
                raise ValueError("Rejected native movement requires an interruption, rejection or unavailable hero")
            if result.status == "no_route_known" and result.destination_reached is not False:
                raise ValueError("A known-route failure cannot claim arrival")
            before = next(hero for hero in self.decision.observation.heroes if hero.handle == action.subject)
            changed = ((result.position is not None and result.position != before.position)
                       or (result.movement is not None and result.movement != before.movement))
            if result.status == "no_route_known" and (self.commands != 0 or changed):
                raise ValueError("Known-route failure requires no native command or movement effect")
            if (changed or result.status == "interaction") and self.accepted_commands == 0:
                raise ValueError("Movement effects require an acknowledged accepted native command")
            if result.position is not None:
                if any(p >= n for p, n in zip(result.position, self.decision.observation.size)):
                    raise ValueError("Movement result outside public map bounds")
                actual = result.position[:2] == (action.x, action.y)
                if result.destination_reached is not actual:
                    raise ValueError("Destination flag must reflect actual position")
            elif result.destination_reached is not None:
                raise ValueError("Cannot claim arrival without observed position")
        else:
            if any(x is not None for x in (result.position, result.movement, result.destination_reached)):
                raise ValueError("Only movement has movement-result fields")
            expected = {"open_hero": "opened", "open_town": "opened", "close": "closed", "end_turn": "turn_ended"}
            if result.status not in (expected[action.kind], "rejected"):
                raise ValueError("Incorrect contextual result")
            target = Context() if action.kind == "close" else Context(action.kind[5:], action.subject) if action.kind.startswith("open_") else self.context
            if action.kind == "end_turn" and result.accepted and (self.commands != 1 or self.accepted_commands != 1):
                raise ValueError("Successful turn ending requires exactly one accepted native command")
            if action.kind == "end_turn" and not result.accepted and self.accepted_commands != 0:
                raise ValueError("Accepted native turn ending cannot be reported as rejected")
            if result.accepted and action.kind != "end_turn" and self.context != target:
                raise ValueError("Navigation result requires matching explicit context change")
            if not result.accepted and self.context != self.decision.observation.context:
                raise ValueError("Rejected navigation cannot have changed context")

    def transitions(self):
        """No targets until boundary finalization; incomplete actions are excluded.

        Successors are subsequent decisions in THIS history, never opponent
        turns. This is environment provenance, not behavior probabilities, value
        estimates or a PPO rollout. A failed record has no usable targets.
        """
        completed = []
        actions = {}
        consumed = {f.decision_seq for f in self._frames if f.kind == "consumed"}
        withdrawn = {f.decision_seq for f in self._frames if f.kind == "superseded" and f.decision_seq not in consumed}
        decisions = [f.seq for f in self._frames if f.kind == "decision" and f.seq not in withdrawn]
        for frame in self._frames:
            if frame.kind == "submitted":
                actions[frame.decision_seq] = frame.action
            elif frame.kind == "result":
                completed.append((frame.decision_seq, actions[frame.decision_seq], frame.seq))
        eligible = self.boundary in ("natural", "cutoff")
        transitions = []
        for index, (decision, action, result) in enumerate(completed):
            successor = next((seq for seq in decisions if seq > result), None)
            terminal_owner = (self.boundary == "natural" and index == len(completed) - 1
                              and not any(seq > decision for seq in consumed))
            if terminal_owner:
                successor = None
            valid = eligible and (successor is not None or terminal_owner)
            reward = (1 if self.terminal == "win" else -1) if valid and terminal_owner else 0 if valid else None
            transitions.append(Transition(self.episode, self.player, decision, action, result, successor,
                                          reward, valid, valid and successor is not None,
                                          self.boundary or "pending"))
        return tuple(transitions)


def finalize_natural(histories, *, engine_outcomes, socket_outcomes, process_exit_code, evidence_ref,
                     histories_complete=(True, True)):
    """Finalize both seats after external evidence checks; never infer from exit alone."""
    if (type(histories) is not tuple or len(histories) != 2 or any(type(h) is not PlayerHistory for h in histories)
            or tuple(h.player for h in histories) != (0, 1) or histories[0].episode != histories[1].episode):
        raise ValueError("Finalization requires both separate player histories in seat order")
    if any(h.boundary is not None or h.terminal != socket_outcomes[h.player] for h in histories):
        raise ValueError("Both unfinalized terminal streams must agree with evidence")
    proof = Verification(histories[0].episode, evidence_ref, engine_outcomes, socket_outcomes, process_exit_code,
                         histories_complete, tuple(h.digest() for h in histories))
    for history in histories:
        history.emit("finalized", verification=proof)
    return proof


def finalize_cutoff(histories, *, evidence_ref, histories_complete, failure_free, shutdown_verified,
                    continuation_limit, continuation_steps):
    """Commit both provisional cutoff prefixes only after external final checks."""
    if (type(histories) is not tuple or len(histories) != 2 or any(type(h) is not PlayerHistory for h in histories)
            or tuple(h.player for h in histories) != (0, 1) or histories[0].episode != histories[1].episode
            or any(h.boundary != "cutoff_pending" for h in histories)):
        raise ValueError("Both players need provisional own-decision cutoffs")
    proof = CutoffVerification(histories[0].episode, evidence_ref, histories_complete, failure_free,
                              shutdown_verified, continuation_limit, continuation_steps,
                              tuple(h.digest() for h in histories))
    for history in histories:
        history.emit("cutoff_finalized", cutoff_verification=proof)
    return proof


def _record(raw, cls):
    if type(raw) is not dict or set(raw) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown fields in {cls.__name__}")
    return dict(raw)


def _array(raw):
    if type(raw) is not list:
        raise ValueError("Expected JSON array")
    return tuple(raw)


def _decode_simple(raw, cls, tuple_fields=()):
    data = _record(raw, cls)
    for name in tuple_fields:
        if data.get(name) is not None:
            data[name] = _array(data[name])
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"Missing or invalid {cls.__name__} fields") from exc


def decode_observation(raw):
    data = _record(raw, Observation)
    for name in ("size", "resources"):
        if name in data:
            data[name] = _array(data[name])
    for name, cls in (("tiles", Tile), ("objects", ObjectView), ("heroes", Hero), ("towns", Town)):
        if name in data:
            data[name] = tuple(_decode_simple(item, cls, ("position",)) for item in _array(data[name]))
    if data.get("window") is not None:
        window = _record(data["window"], Window)
        window["context"] = _decode_simple(window.get("context"), Context)
        window["army"] = tuple(_decode_simple(item, Stack) for item in _array(window.get("army")))
        for name in ("primary_skills", "buildings"):
            if window.get(name) is not None:
                window[name] = _array(window[name])
        try:
            data["window"] = Window(**window)
        except TypeError as exc:
            raise ValueError("Incomplete window") from exc
    try:
        return Observation(**data)
    except TypeError as exc:
        raise ValueError("Incomplete observation") from exc


def decode_notice(raw):
    """Allowlisted JSON notice; unused pools/raw text/engine IDs are rejected."""
    data = _record(raw, Notice)
    data["text"] = tuple(_decode_simple(token, NoticeToken) for token in _array(data.get("text")))
    data["components"] = tuple(_decode_simple(component, NoticeComponent)
                               for component in _array(data.get("components")))
    try:
        return Notice(**data)
    except TypeError as exc:
        raise ValueError("Incomplete informational notice") from exc


def encode_frame(frame):
    if type(frame) is not Frame:
        raise ValueError("Expected typed contextual frame")
    # JSON conversion changes immutable tuples to arrays without sharing mutable data.
    return json.loads(json.dumps(asdict(frame)))


def decode_frame(raw):
    data = _record(raw, Frame)
    if not {"schema", "episode", "player", "seq", "day", "kind"} <= data.keys():
        raise ValueError("History JSON requires explicit schema and base fields")
    if data.get("observation") is not None:
        data["observation"] = decode_observation(data["observation"])
    for name, cls in (("context", Context), ("action", Action), ("result", Result)):
        if data.get(name) is not None:
            data[name] = _decode_simple(data[name], cls, ("position",) if name == "result" else ())
    if data.get("event") is not None:
        event = _record(data["event"], VisibleEvent)
        if event.get("notice") is not None:
            event["notice"] = decode_notice(event["notice"])
        data["event"] = _decode_simple(event, VisibleEvent)
    if data.get("offers") is not None:
        offer = _record(data["offers"], Offers)
        offer["context"] = _decode_simple(offer.get("context"), Context)
        for name in ("size", "hero_refs", "town_refs"):
            if name in offer:
                offer[name] = _array(offer[name])
        try:
            data["offers"] = Offers(**offer)
        except TypeError as exc:
            raise ValueError("Incomplete offers") from exc
    if data.get("verification") is not None:
        data["verification"] = _decode_simple(data["verification"], Verification,
                                              ("engine_outcomes", "socket_outcomes", "histories_complete", "history_sha256"))
    if data.get("cutoff_verification") is not None:
        data["cutoff_verification"] = _decode_simple(data["cutoff_verification"], CutoffVerification,
                                                     ("histories_complete", "continuation_steps", "history_sha256"))
    try:
        return Frame(**data)
    except TypeError as exc:
        raise ValueError("Incomplete contextual frame") from exc
