"""Small randomly initialized recurrent actor-critic for player-history v1.

This module is an offline component, not a live bridge adapter or a trainer.
The typed contract still depends on an audited producer for visibility truth.
Use ``forward_step`` to preserve autograd across history for future BPTT; use
one ``PlayerSession`` per player for checked, no-grad inference.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from math import copysign, log1p, sqrt
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .actions import (
    ACTION_TYPES, ARGUMENT_CATEGORIES, ARGUMENT_SLOTS, CandidateSet,
    movement_candidates,
)
from .contract import (
    BOARD_SIZE, CREATURE_TYPES, OBJECT_SUBTYPES, OBJECT_TYPES, OWNERS,
    TERRAIN_TYPES, Frame, HistoryCursor, View,
)


@dataclass(frozen=True)
class PolicyValue:
    """Nine masked logits and a scalar expected terminal-return prediction."""

    logits: Tensor
    value: Tensor

    def distribution(self) -> Categorical:
        # Contract validation guarantees at least end-turn is enabled.
        return Categorical(logits=self.logits)


@dataclass(frozen=True)
class StepOutput:
    state: Tensor
    decision: PolicyValue | None


class RecurrentActorCritic(nn.Module):
    """Shared encoder/GRU with a variable candidate scorer and value head.

    The fixed spatial layout is the authored 20x20x1 scenario. Unknown tiles
    have zero current-view features; visible object embeddings are summed at
    their reported tile. Every view rebuilds this grid from scratch. Only the
    recurrent state can retain earlier observations. No object handle, player
    color, episode identity, sequence number, or decision reference is encoded.
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        # Index zero is absent/unobserved; real categories are offset by one.
        self.terrain = nn.Embedding(TERRAIN_TYPES + 1, 8, padding_idx=0)
        self.object_kind = nn.Embedding(OBJECT_TYPES + 1, 8, padding_idx=0)
        self.object_subtype = nn.Embedding(OBJECT_SUBTYPES + 1, 4, padding_idx=0)
        self.owner = nn.Embedding(len(OWNERS) + 1, 3, padding_idx=0)
        # Fuse attributes before aggregation so co-located objects retain the
        # binding between their type, subtype and relative ownership.
        self.object_encoder = nn.Sequential(nn.Linear(15, 15), nn.Tanh())
        self.creature = nn.Embedding(CREATURE_TYPES + 1, 4, padding_idx=0)
        # 8 terrain + 15 object + visibility/block/visit + own hero/town.
        self.map_encoder = nn.Sequential(
            nn.Conv2d(28, 16, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.Tanh(),
            nn.Flatten(), nn.Linear(16 * 5 * 5, 32), nn.Tanh(),
        )
        # Seven resources, movement/current maximum, mine count, hero flag,
        # plus seven fixed army slots of (four learned features, log count).
        self.own_encoder = nn.Sequential(nn.Linear(11 + 7 * 5, 32), nn.Tanh())
        self.frame_kind = nn.Embedding(4, 4)
        self.action = nn.Embedding(10, 4, padding_idx=0)
        self.result = nn.Embedding(3, 2, padding_idx=0)
        # Map + own + kind/action/result + day + view flag + nine mask flags.
        self.gru = nn.GRUCell(32 + 32 + 4 + 4 + 2 + 1 + 1 + 9, hidden_size)
        self.candidate_type = nn.Embedding(ACTION_TYPES, 8)
        self.argument_slot = nn.Embedding(ARGUMENT_SLOTS, 4)
        self.argument_kind = nn.Embedding(3, 3)
        self.argument_category = nn.Embedding(ARGUMENT_CATEGORIES + 1, 8, padding_idx=0)
        # Bind the action type, argument slot/kind, categorical value and numeric
        # value before pooling. This avoids losing which value belongs to which
        # parameter when the producer presents several arguments.
        self.argument_encoder = nn.Sequential(nn.Linear(8 + 4 + 3 + 8 + 4, 16), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(8 + 16, 32), nn.Tanh())
        self.policy = nn.Linear(hidden_size, 32)
        self.value = nn.Linear(hidden_size, 1)

    @property
    def device(self) -> torch.device:
        return self.terrain.weight.device

    def initial_state(self) -> Tensor:
        return self.terrain.weight.new_zeros(self.hidden_size)

    def _long(self, values) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _float(self, values) -> Tensor:
        return self.terrain.weight.new_tensor(values)

    def _encode_view(self, view: View) -> tuple[Tensor, Tensor]:
        width, height, _ = BOARD_SIZE
        terrain = [0] * (width * height)
        flags = [[0.0] * 5 for _ in terrain]
        # Stable semantic ordering also makes floating-point reductions invariant
        # to wire order and opaque handle renaming.
        for tile in sorted(view.tiles, key=lambda t: t.position):
            x, y, _ = tile.position
            index = y * width + x
            terrain[index] = tile.terrain + 1
            flags[index][:3] = [1.0, float(tile.blocked), float(tile.visitable)]
        if view.own.hero is not None:
            x, y, _ = view.own.hero
            flags[y * width + x][3] = 1.0
        for x, y, _ in sorted(view.own.towns):
            flags[y * width + x][4] = 1.0

        objects = sorted(view.objects,
                         key=lambda o: (o.position, o.kind, o.subtype, o.owner))
        object_grid = self.terrain.weight.new_zeros(width * height, 15)
        if objects:
            embeddings = torch.cat((
                self.object_kind(self._long([o.kind + 1 for o in objects])),
                self.object_subtype(self._long([o.subtype + 1 for o in objects])),
                self.owner(self._long([OWNERS.index(o.owner) + 1 for o in objects])),
            ), dim=-1)
            locations = self._long([o.position[1] * width + o.position[0] for o in objects])
            object_grid = object_grid.index_add(0, locations, self.object_encoder(embeddings))
        grid = torch.cat((self.terrain(self._long(terrain)), object_grid,
                          self._float(flags)), dim=-1)
        grid = grid.reshape(height, width, 28).permute(2, 0, 1).unsqueeze(0)
        map_features = self.map_encoder(grid).squeeze(0)

        own = view.own
        creatures, counts = [0] * 7, [0.0] * 7
        for stack in sorted(own.army, key=lambda s: s.slot):
            creatures[stack.slot] = stack.creature + 1
            counts[stack.slot] = log1p(stack.count)
        army = torch.cat((self.creature(self._long(creatures)),
                          self._float(counts).unsqueeze(-1)), dim=-1).flatten()
        scalars = [log1p(v) for v in (*own.resources, own.movement,
                                    own.movement_max, own.mines)]
        scalars.append(float(own.hero is not None))
        own_features = self.own_encoder(torch.cat((self._float(scalars), army)))
        return map_features, own_features

    def encode_frame(self, frame: Frame) -> Tensor:
        """Encode only the allowlisted semantics, without consulting old state."""
        if type(frame) is not Frame:
            raise ValueError("Expected a typed Frame, never a raw engine message")
        if frame.view is None:
            map_features = own_features = self.terrain.weight.new_zeros(32)
        else:
            map_features, own_features = self._encode_view(frame.view)
        kind = ("view", "decision", "action", "result").index(frame.kind)
        action = 0 if frame.action is None else frame.action + 1
        result = 0 if frame.accepted is None else 1 + int(frame.accepted)
        numeric = [log1p(frame.day), float(frame.view is not None)]
        numeric.extend(float(v) for v in (frame.mask or (False,) * 9))
        return torch.cat((map_features, own_features,
                          self.frame_kind(self._long(kind)),
                          self.action(self._long(action)),
                          self.result(self._long(result)), self._float(numeric)))

    def _validate_state(self, state: Tensor) -> None:
        if (not isinstance(state, Tensor) or state.shape != (self.hidden_size,)
                or state.device != self.device or state.dtype != self.terrain.weight.dtype):
            raise ValueError("Memory must match the model's shape, device and dtype")

    def score_candidates(self, state: Tensor, candidates: CandidateSet) -> Tensor:
        """Return one contextual logit per permitted candidate, in supplied order.

        Candidate keys are excluded from features. Vocabulary codes identify
        player-exposed controls; no game-option catalog or utility is embedded
        here. History v1 still records movement only; accepting a typed candidate
        does not claim a live adapter exists for that action or its result.
        """
        self._validate_state(state)
        if type(candidates) is not CandidateSet:
            raise ValueError("Expected a typed CandidateSet")
        representations = []
        for candidate in candidates.candidates:
            type_features = self.candidate_type(self._long(candidate.type_id))
            arguments = []
            for argument in sorted(candidate.arguments, key=lambda a: a.slot):
                category, numeric = 0, [0.0] * 4
                if argument.kind == "category":
                    category = argument.value + 1
                elif argument.kind == "scalar":
                    numeric[0] = copysign(log1p(abs(argument.value)), argument.value)
                else:
                    numeric[1:] = [value / max(extent - 1, 1)
                                   for value, extent in zip(argument.value, BOARD_SIZE)]
                kind = ("category", "scalar", "position").index(argument.kind)
                arguments.append(self.argument_encoder(torch.cat((
                    type_features, self.argument_slot(self._long(argument.slot)),
                    self.argument_kind(self._long(kind)),
                    self.argument_category(self._long(category)), self._float(numeric),
                ))))
            pooled = (torch.stack(arguments).sum(0) if arguments
                      else self.terrain.weight.new_zeros(16))
            representations.append(self.candidate_encoder(torch.cat((type_features, pooled))))
        context = torch.tanh(self.policy(state))
        return torch.stack(representations).mv(context) / sqrt(context.numel())

    def forward_step(self, frame: Frame, state: Tensor) -> StepOutput:
        """One differentiable history step; caller owns chronology and boundaries.

        This low-level API deliberately does not detach memory. Use a separate
        HistoryCursor for each player, or PlayerSession for checked inference.
        No policy/value output is produced for non-decision history events.
        """
        self._validate_state(state)
        state = self.gru(self.encode_frame(frame), state)
        if frame.kind != "decision":
            return StepOutput(state, None)
        candidates = movement_candidates(frame.mask)
        scores = self.score_candidates(state, candidates)
        enabled = self._long([index for index, allowed in enumerate(frame.mask) if allowed])
        logits = scores.new_full((9,), -torch.inf).scatter(0, enabled, scores)
        return StepOutput(state, PolicyValue(logits, self.value(state).squeeze(-1)))


class PlayerSession:
    """Checked inference memory for exactly one episode/player at a time.

    A turn or rollout chunk is not an episode boundary. ``finish`` explicitly
    closes a stream without judging its outcome; reward validation belongs to
    the trajectory layer. A fresh episode can then start with ``reset``.
    The shared model must remain fixed during a rollout, as specified in README.
    """

    def __init__(self, model: RecurrentActorCritic, episode: str, player: int):
        self.model = model
        self.cursor = HistoryCursor(episode, player)
        self._state: Tensor | None = model.initial_state()
        self._used_episodes = {episode}

    @property
    def state(self) -> Tensor:
        if self._state is None:
            raise ValueError("Session has finished; reset for a new episode")
        return self._state.detach().clone()

    def consume(self, frame: Frame) -> PolicyValue | None:
        if self._state is None:
            raise ValueError("Session has finished; reset for a new episode")
        # Failed validation/inference must not partially advance the stream.
        candidate = copy(self.cursor)
        candidate.consume(frame)
        with torch.no_grad():
            output = self.model.forward_step(frame, self._state)
        self.cursor, self._state = candidate, output.state
        return output.decision

    def consume_chunk(self, frames: Iterable[Frame]) -> tuple[PolicyValue, ...]:
        decisions = []
        for frame in frames:
            decision = self.consume(frame)
            if decision is not None:
                decisions.append(decision)
        return tuple(decisions)

    def finish(self) -> None:
        if self._state is None:
            raise ValueError("Session has already finished")
        self._state = None

    def reset(self, episode: str, player: int | None = None) -> None:
        if self._state is not None:
            raise ValueError("Finish the prior episode before resetting memory")
        if episode in self._used_episodes:
            raise ValueError("Reset requires a fresh episode identity")
        cursor = HistoryCursor(episode, self.cursor.player if player is None else player)
        self.cursor, self._state = cursor, self.model.initial_state()
        self._used_episodes.add(episode)
