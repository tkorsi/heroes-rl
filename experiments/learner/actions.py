"""Versioned candidate-scoring boundary, without a catalog of game choices.

The producer supplies only choices exposed by the player's legitimate interface.
Type/argument codes are opaque categorical identifiers, not strategic meanings.
The movement adapter is the only live-interface vocabulary specified here.
Engine object IDs and arbitrary local entity handles must never be encoded as
category/scalar arguments. They remain routing metadata; target features must
come from legitimate player observations. Shape validation cannot attest this.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .contract import integer, position


ACTION_SCHEMA = "heroes-action-candidates-v1"
ACTION_TYPES = 256
ARGUMENT_SLOTS = 64
ARGUMENT_CATEGORIES = 4096


@dataclass(frozen=True)
class ActionArgument:
    slot: int
    kind: Literal["category", "scalar", "position"]
    value: int | float | tuple[int, int, int]

    def __post_init__(self):
        integer(self.slot, 0, ARGUMENT_SLOTS - 1, "argument slot")
        if self.kind == "category":
            integer(self.value, 0, ARGUMENT_CATEGORIES - 1, "argument category")
        elif self.kind == "scalar":
            if type(self.value) not in (int, float) or not math.isfinite(self.value):
                raise ValueError("Scalar argument must be finite")
        elif self.kind == "position":
            position(self.value)
        else:
            raise ValueError("Unknown action argument type")


@dataclass(frozen=True)
class ActionCandidate:
    key: str
    type_id: int
    arguments: tuple[ActionArgument, ...] = ()

    def __post_init__(self):
        if type(self.key) is not str or not self.key:
            raise ValueError("Candidate needs an opaque decision-local routing key")
        integer(self.type_id, 0, ACTION_TYPES - 1, "action type")
        if (type(self.arguments) is not tuple
                or any(type(a) is not ActionArgument for a in self.arguments)):
            raise ValueError("Action arguments must be typed and immutable")
        if len({a.slot for a in self.arguments}) != len(self.arguments):
            raise ValueError("Duplicate action argument slot")


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[ActionCandidate, ...]
    schema: str = ACTION_SCHEMA

    def __post_init__(self):
        if self.schema != ACTION_SCHEMA:
            raise ValueError("Unsupported action contract; explicit migration is required")
        if (type(self.candidates) is not tuple or not self.candidates
                or any(type(c) is not ActionCandidate for c in self.candidates)):
            raise ValueError("Expected a nonempty immutable set of permitted candidates")
        if len({c.key for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidate routing key")
        signatures = {(c.type_id, tuple(sorted(c.arguments, key=lambda a: a.slot)))
                      for c in self.candidates}
        if len(signatures) != len(self.candidates):
            raise ValueError("Duplicate semantic candidates would distort behavior probabilities")


def movement_candidates(mask: tuple[bool, ...]) -> CandidateSet:
    """Exact current support: move(type 0, direction slot 0) or end-turn(type 1).

    Direction codes retain the existing N, NE, E, SE, S, SW, W, NW action order.
    They define controls, not destinations, utility, or a scripted strategy.
    """
    if (type(mask) is not tuple or len(mask) != 9
            or any(type(v) is not bool for v in mask) or not mask[8]):
        raise ValueError("Expected the movement prototype's nine boolean attempt mask")
    return CandidateSet(tuple(
        ActionCandidate(str(action), 0, (ActionArgument(0, "category", action),))
        if action < 8 else ActionCandidate("8", 1)
        for action, enabled in enumerate(mask) if enabled
    ))
