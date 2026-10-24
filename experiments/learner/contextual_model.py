"""Randomly initialized contextual actor-critic; no collection or optimization.

The codec consumes strict player-local chronology. Routing references resolve only
against the supplied current/earlier permitted snapshot; their spelling is never
encoded. Current map grids are rebuilt from observations. Only the learned GRU
retains earlier information: there is no accumulated last-seen world cache.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from math import copysign, log1p, sqrt

import torch
from torch import Tensor, nn

from . import contextual as c

MODEL_SCHEMA = "heroes-context-model-v1"
DISTRIBUTION_SCHEMA = "heroes-context-policy-v1"
ARMIES_MODEL_SCHEMA = "heroes-context-model-v2"
ARMIES_DISTRIBUTION_SCHEMA = "heroes-context-policy-v2"
_ACTIONS = ("move_to", "open_hero", "open_town", "end_turn", "close", "open_panel",
            "open_building", "construct", "back", "select_recruit", "recruit", "transfer_stack")
# Verification, command identifiers and technical boundaries are not observations.
_METADATA = frozenset(("issued", "cutoff", "cutoff_finalized", "finalized", "failure", "superseded"))


@dataclass(frozen=True)
class ModelConfig:
    schema: str = MODEL_SCHEMA
    hidden_size: int = 64
    feature_size: int = 32
    max_map_side: int = 256

    def __post_init__(self):
        if self.schema not in (MODEL_SCHEMA, ARMIES_MODEL_SCHEMA):
            raise ValueError("Unsupported contextual model schema")
        for name in ("hidden_size", "feature_size", "max_map_side"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer model dimensions required")
        if self.max_map_side > 256:
            raise ValueError("This bounded model supports map sides through 256")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Exact versioned model configuration required")
        return cls(**value)


def model_schema_for(profile_or_schema):
    return ARMIES_MODEL_SCHEMA if c.profile_for(profile_or_schema).army_transfer else MODEL_SCHEMA


def distribution_schema_for(profile_or_schema):
    return ARMIES_DISTRIBUTION_SCHEMA if c.profile_for(profile_or_schema).army_transfer else DISTRIBUTION_SCHEMA


def weights_sha256(model):
    """Bind architecture/config and named tensors without NumPy or pickle."""
    digest = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().clone()
        header = json.dumps((name, str(value.dtype), tuple(value.shape)), separators=(",", ":")).encode()
        # Tensor's byte-list conversion runs in C; iterating untyped_storage
        # through Python is hundreds of milliseconds even for this small model.
        data = bytes(value.reshape(-1).view(torch.uint8).tolist())
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def _node(tag, *children):
    return ("node", tag, tuple(children))


def _number(value):
    return ("number", float(value))


def _text(value):
    return ("text", value)


def _set(tag, children):
    # No opaque-ref order, hash order or engine allocation order becomes a feature.
    return _node(tag, *sorted(children, key=repr))


def _subject(observation, ref):
    if ref is None or observation is None:
        return _node("unresolved_reference")
    for entry in (*observation.heroes, *observation.towns, *observation.objects):
        if entry.handle == ref:
            return _semantic(entry)
    return _node("unresolved_reference")


def _selection(observation, ref):
    if observation is None or observation.window is None or observation.window.town_panel is None:
        return _node("unresolved_selection")
    panel = observation.window.town_panel
    for entry in (*panel.controls, *panel.buildings,
                  *((panel.building,) if panel.building else ()),
                  *(panel.recruitment.offers if panel.recruitment else ())):
        if entry.ref == ref:
            return _semantic(entry)
    return _node("unresolved_selection")


def _action(action, observation):
    # The binding is the held decision's snapshot, never a newly queried subject.
    children = [_text(action.kind), _node("decision_subject", _subject(observation, action.subject))]
    for key in ("x", "y", "count"):
        value = getattr(action, key)
        children.append(_node(key, _semantic(value)))
    children.extend((_node("panel", _semantic(action.panel)),
                     _node("decision_selection", _army_slot(observation, action.selection, receiving=False)
                           if action.kind == "transfer_stack" else _selection(observation, action.selection))))
    if action.destination is not None:
        if action.kind == "transfer_stack":
            children.append(_node("destination", _army_slot(observation, action.destination, receiving=True)))
        else:
            panel = observation.window.town_panel.recruitment
            if panel is None or panel.destination.ref != action.destination:
                raise ValueError("Action destination has no displayed semantic binding")
            children.append(_node("destination", _semantic(panel.destination)))
    return _node("action", *children)


def _army_slot(observation, ref, *, receiving):
    panel = observation.window.town_panel if observation and observation.window else None
    exchange = panel.armies if panel else None
    army = (exchange.destination if receiving else exchange.source) if exchange else None
    slot = next((slot for slot in army.slots if slot.ref == ref), None) if army else None
    if slot is None:
        raise ValueError("Army slot has no current displayed semantic binding")
    return _node("army_slot", _node("role", _text(army.role)), _node("owner_subject", _subject(observation, army.subject)),
                 _node("requires_nonempty", _semantic(army.requires_nonempty)), _semantic(slot))


def _semantic(value):
    """Lossless field selection before learned compression; no reference IDs.

    Static definition numbers/text identifiers are categorical bytes learned from
    random initialization, not ordinal utility or pretrained text embeddings.
    Ordered sequences use a separate recurrent reduction (notice substitutions).
    """
    if value is None:
        return _node("absent")
    if type(value) is bool:
        return _text("true" if value else "false")
    if type(value) in (float, int):
        return _number(value)
    if type(value) is str:
        return _text(value)
    if type(value) is tuple:
        return ("ordered", "tuple", tuple(_semantic(item) for item in value))
    if type(value) is c.RecruitmentPanel:
        cards = [_node("card", _semantic(card), _semantic(card.ref == value.selected_offer_ref)) for card in value.offers]
        return _node("recruitment", _set("cards", cards), _node("stock", _number(value.stock)),
                     _node("unit_cost", _semantic(value.unit_cost)), _node("slider_max", _number(value.slider_max)),
                     _semantic(value.destination))
    if type(value) is c.Notice:
        return _node("notice", _text(value.display_mode),
                     ("ordered", "notice_text", tuple(_semantic(token) for token in value.text)),
                     _set("components", [_semantic(component) for component in value.components]))
    if is_dataclass(value):
        children = []
        for field in fields(value):
            name, item = field.name, getattr(value, field.name)
            # This additive v4 field must not add even an "absent" node to old
            # observations: archived v1 values/probabilities remain exact.
            if type(value) is c.TownPanel and name == "armies" and item is None:
                continue
            if name in ("handle", "ref", "control_ref", "selected_offer_ref", "subject", "selection", "schema"):
                continue
            if name in ("creature", "terrain", "road", "kind", "subtype", "object_kind", "object_subtype", "faction") and type(item) is int:
                encoded = _text(f"category:{item}")
            elif name in ("children", "controls", "buildings", "army") and type(item) is tuple:
                encoded = _set(name, [_semantic(entry) for entry in item])
            else:
                encoded = _semantic(item)
            children.append(_node(name, encoded))
        return _node(type(value).__name__, *children)
    raise ValueError("Unsupported model semantic payload")


@dataclass(frozen=True)
class ContextualInput:
    frame: c.Frame
    semantics: tuple | None
    # These are the frame's CURRENT observation, never an accumulated map cache.
    observation: c.Observation | None = None


class ContextualEncoder:
    """Chronology checker plus temporary routing bindings, with no neural state."""

    def __init__(self, episode, player, schema=c.SCHEMA):
        self.history = c.PlayerHistory(episode, player, schema=schema)
        self.observation = None

    def encode(self, frame):
        candidate = copy(self.history)
        candidate._frames = list(self.history.frames)
        candidate.append(frame)
        basis = self.history.decision.observation if self.history.decision is not None else self.observation
        kind = frame.kind
        payload = None
        if kind not in _METADATA:
            children = [_node("day", _number(frame.day)), _text(kind)]
            if frame.action is not None:
                children.append(_action(frame.action, basis))
            if frame.event is not None:
                children.append(_semantic(frame.event))
                if frame.event.subject is not None:
                    children.append(_node("prior_observation_subject", _subject(self.observation, frame.event.subject)))
            if frame.result is not None:
                children.append(_semantic(frame.result))
            if frame.context is not None:
                children.append(_semantic(frame.context))
                children.append(_node("prior_observation_subject", _subject(self.observation, frame.context.subject)))
            if frame.accepted is not None:
                children.append(_node("accepted", _semantic(frame.accepted)))
            if frame.outcome is not None:
                children.append(_node("observed_outcome", _text(frame.outcome)))
            # Consumer layer, revisions, sequence numbers, evidence and command
            # counters are deliberately absent. A consumed-action marker remains.
            payload = _node("frame", *children)
        result = ContextualInput(frame, payload, frame.observation)
        self.history = candidate
        if frame.observation is not None:
            self.observation = frame.observation
        return result


def encode_history(frames):
    frames = tuple(frames)
    if not frames:
        return ()
    first = frames[0]
    encoder = ContextualEncoder(first.episode, first.player, first.schema)
    return tuple(encoder.encode(frame) for frame in frames)


@dataclass(frozen=True)
class DecisionSupport:
    action_schema: str
    context: c.Context
    offers: c.Offers
    distribution_schema: str = DISTRIBUTION_SCHEMA

    def __post_init__(self):
        if (self.distribution_schema != distribution_schema_for(self.action_schema) or type(self.offers) is not c.Offers
                or self.action_schema != self.offers.schema or self.context != self.offers.context):
            raise ValueError("Exact contextual policy support required")


@dataclass(frozen=True)
class ConditionalProbability:
    name: str
    selected_index: int
    support_size: int
    log_prob: Tensor


ConditionalLogProbability = ConditionalProbability


@dataclass(frozen=True)
class ActionEvaluation:
    log_prob: Tensor
    conditionals: tuple[ConditionalProbability, ...]


@dataclass(frozen=True)
class PolicyValue:
    value: Tensor
    distribution: ContextualDistribution


@dataclass(frozen=True)
class StepOutput:
    state: Tensor
    decision: PolicyValue | None


class ContextualActorCritic(nn.Module):
    """Current-view spatial encoder and typed semantics share one learned GRU.

    No object/choice routing handle, color, episode, revision or proof is encoded.
    The tree encoder binds fields within entities before symmetric pooling;
    notice operations retain order. Text bytes have randomly initialized learned
    embeddings; there are no language-model weights or built-in strategy scores.
    """

    def __init__(self, config=ModelConfig()):
        super().__init__()
        if type(config) is not ModelConfig:
            raise ValueError("Expected ModelConfig")
        self.config = config
        f, h = config.feature_size, config.hidden_size
        self.byte = nn.Embedding(257, 8, padding_idx=0)
        self.text_gru = nn.GRU(8, f, batch_first=True)
        self.number = nn.Sequential(nn.Linear(2, f), nn.Tanh())
        self.tree = nn.Sequential(nn.Linear(2 * f, f), nn.Tanh())
        self.ordered = nn.GRU(f, f, batch_first=True)
        self.terrain = nn.Embedding(257, 8, padding_idx=0)
        self.road = nn.Embedding(257, 4, padding_idx=0)
        self.object_encoder = nn.Sequential(nn.Linear(32 * 2 + 3, 16), nn.Tanh())
        self.map_cnn = nn.Sequential(nn.Conv2d(33, 16, 3, padding=1), nn.Tanh(),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.Tanh())
        self.map_global = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, f), nn.Tanh())
        self.gru = nn.GRUCell(3 * f, h)
        self.policy_query = nn.Linear(h, f)
        self.value = nn.Linear(h, 1)
        self.coordinate = nn.Sequential(nn.Linear(h + f + 18, f), nn.Tanh(), nn.Linear(f, 1))
        self.quantity = nn.Sequential(nn.Linear(h + f + 4, f), nn.Tanh(), nn.Linear(f, 2))

    @property
    def device(self):
        return self.byte.weight.device

    def initial_state(self):
        return self.byte.weight.new_zeros(self.config.hidden_size)

    def _float(self, values):
        return self.byte.weight.new_tensor(values)

    def _long(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _encode_tree(self, tree, cache):
        if tree in cache:
            return cache[tree]
        kind = tree[0]
        if kind == "text":
            raw = tree[1].encode("utf-8")
            if not raw:
                result = self.byte.weight.new_zeros(self.config.feature_size)
            else:
                embedding = self.byte(self._long([value + 1 for value in raw])).unsqueeze(0)
                result = self.text_gru(embedding)[1][0, 0]
        elif kind == "number":
            value = tree[1]
            result = self.number(self._float((copysign(log1p(abs(value)), value), float(value == 0))))
        else:
            tag, children = tree[1:]
            encodings = [self._encode_tree(child, cache) for child in children]
            if not encodings:
                pooled = self.byte.weight.new_zeros(self.config.feature_size)
            elif kind == "ordered":
                pooled = self.ordered(torch.stack(encodings).unsqueeze(0))[1][0, 0]
            else:
                pooled = torch.stack(encodings).sum(0) / sqrt(len(encodings))
            result = self.tree(torch.cat((self._encode_tree(_text(tag), cache), pooled)))
        cache[tree] = result
        return result

    def _encode_observation(self, observation, cache):
        width, height, _ = observation.size
        if width > self.config.max_map_side or height > self.config.max_map_side:
            raise ValueError("Map exceeds declared model configuration")
        terrain, road = [0] * (width * height), [0] * (width * height)
        flags = [[0.] * 5 for _ in terrain]
        for tile in observation.tiles:
            index = tile.position[1] * width + tile.position[0]
            terrain[index], road[index] = tile.terrain + 1, tile.road + 1
            flags[index][:3] = (1., float(tile.blocked), float(tile.visitable))
        for hero in observation.heroes:
            flags[hero.position[1] * width + hero.position[0]][3] += 1.
        for town in observation.towns:
            flags[town.position[1] * width + town.position[0]][4] += 1.
        objects = sorted(observation.objects, key=lambda obj: (obj.position, obj.kind, obj.subtype, obj.owner))
        object_grid = self.byte.weight.new_zeros(width * height, 16)
        if objects:
            # Static content IDs are categorical bit patterns, not scalar ranks.
            attributes = [[float(value >> bit & 1) for value in (obj.kind, obj.subtype) for bit in range(32)]
                          + [float(obj.owner == owner) for owner in ("self", "opponent", "neutral")] for obj in objects]
            indices = self._long([obj.position[1] * width + obj.position[0] for obj in objects])
            object_grid = object_grid.index_add(0, indices, self.object_encoder(self._float(attributes)))
        grid = torch.cat((self.terrain(self._long(terrain)), self.road(self._long(road)), object_grid, self._float(flags)), -1)
        cells = self.map_cnn(grid.reshape(height, width, 33).permute(2, 0, 1).unsqueeze(0))
        global_map = self.map_global(cells)[0]
        own = _node("observation", _node("size", _semantic(observation.size)),
                    _node("resources", _semantic(observation.resources)),
                    _set("heroes", [_semantic(hero) for hero in observation.heroes]),
                    _set("towns", [_semantic(town) for town in observation.towns]),
                    _semantic(observation.window),
                    _node("current_window_subject", _subject(observation, observation.context.subject)))
        return global_map, self._encode_tree(own, cache), cells[0].permute(1, 2, 0).reshape(-1, 16)

    def forward_step(self, frame, state):
        if type(frame) is not ContextualInput:
            raise ValueError("Use the checked contextual history codec")
        if self.config.schema != model_schema_for(frame.frame.schema):
            raise ValueError("Model version does not match this explicit history profile")
        if (not isinstance(state, Tensor) or state.shape != (self.config.hidden_size,)
                or state.device != self.device or state.dtype != self.byte.weight.dtype):
            raise ValueError("State must match model dimensions, dtype and device")
        cache = {}
        zero = self.byte.weight.new_zeros(self.config.feature_size)
        if frame.semantics is None:
            return StepOutput(state, None)
        if frame.observation is None:
            map_features = own = zero
            cells = None
        else:
            map_features, own, cells = self._encode_observation(frame.observation, cache)
        state = self.gru(torch.cat((self._encode_tree(frame.semantics, cache), map_features, own)), state)
        decision = None
        if frame.frame.kind == "decision":
            distribution = ContextualDistribution(self, state, frame.frame.observation, frame.frame.offers, cells, cache)
            decision = PolicyValue(self.value(state).squeeze(-1), distribution)
        return StepOutput(state, decision)


class ContextualDistribution:
    """Typed autoregressive distribution, exact support and chosen-path log P.

    Quantity uses a balanced binary interval tree. At most 31 learned choices
    represent every legal int32 count. There is deliberately no entropy method:
    exact joint count entropy would require visiting every leaf. PPO starts with
    entropy coefficient zero, not a selected-path value mislabeled as entropy.
    """

    def __init__(self, model, state, observation, offers, cells, cache):
        self.model, self.state, self.observation, self.offers = model, state, observation, offers
        self.cells, self.cache = cells, cache
        self.support = DecisionSupport(offers.schema, offers.context, offers, distribution_schema_for(offers.schema))

    def _features(self, semantic):
        return self.model._encode_tree(semantic, self.cache)

    def _logits(self, semantics):
        candidates = torch.stack([self._features(item) for item in semantics])
        return candidates.mv(torch.tanh(self.model.policy_query(self.state))) / sqrt(self.model.config.feature_size)

    def _kinds(self):
        offers, context = self.offers, self.offers.context
        if context.kind == "map":
            return tuple(kind for kind in _ACTIONS if kind == "end_turn"
                         or kind in ("move_to", "open_hero") and offers.hero_refs or kind == "open_town" and offers.town_refs)
        if offers.panel is None:
            return ("close",)
        if context.panel == "overview":
            kinds = ("close", "open_panel") if offers.panel.controls else ("close",)
            return kinds + (("transfer_stack",) if offers.panel.armies and offers.panel.armies.choices() else ())
        if context.panel == "hall":
            return ("open_building", "back") if offers.panel.buildings else ("back",)
        if context.panel == "building":
            return ("construct", "back") if offers.panel.building.status == "allowed" else ("back",)
        recruitment = offers.panel.recruitment
        return ("back", "select_recruit", "recruit") if recruitment.slider_max else ("back", "select_recruit")

    def _walk(self, action=None, generator=None):
        if action is not None and not self.offers.permits(action):
            raise ValueError("Action is outside this exact behavior support")
        trace = []

        def choose(name, logits, selected=None):
            log_probs = torch.log_softmax(logits, 0)
            if selected is None:
                selected = int(torch.multinomial(log_probs.detach().exp(), 1, generator=generator).item())
            trace.append(ConditionalProbability(name, selected, logits.numel(), log_probs[selected]))
            return selected

        kinds = self._kinds()
        kind = kinds[choose("kind", self._logits([_node("action_kind", _text(kind)) for kind in kinds]),
                            kinds.index(action.kind) if action else None)]
        subject = panel = selection = destination = x = y = count = None
        if kind in ("move_to", "open_hero", "open_town"):
            refs = self.offers.town_refs if kind == "open_town" else self.offers.hero_refs
            semantics = [_node(kind, _subject(self.observation, ref)) for ref in refs]
            index = choose("subject", self._logits(semantics), refs.index(action.subject) if action else None)
            subject = refs[index]
            if kind == "move_to":
                width, height, _ = self.offers.size
                positions = self.model._float([(px / max(width - 1, 1), py / max(height - 1, 1))
                                               for py in range(height) for px in range(width)])
                target = self._features(semantics[index]).expand(width * height, -1)
                state = self.state.expand(width * height, -1)
                logits = self.model.coordinate(torch.cat((state, target, self.cells, positions), -1)).squeeze(-1)
                index = choose("coordinate", logits, action.y * width + action.x if action else None)
                x, y = index % width, index // width
        elif kind == "open_panel":
            controls = self.offers.panel.controls
            # There is one semantic hall action even if malformed support repeats
            # an equivalent hall control; never double-count an action's mass.
            hall_count = sum(control.panel == "hall" for control in controls)
            if hall_count > 1:
                raise ValueError("Duplicate semantic hall controls")
            selected = next((i for i, control in enumerate(controls) if action and control.panel == action.panel
                             and (control.panel == "hall" or control.ref == action.selection)), None)
            index = choose("control", self._logits([_semantic(control) for control in controls]), selected)
            control = controls[index]
            subject, panel = self.offers.context.subject, control.panel
            selection = control.ref if panel == "recruitment" else None
        elif kind in ("open_building", "select_recruit"):
            choices = self.offers.panel.buildings if kind == "open_building" else self.offers.panel.recruitment.offers
            index = choose("selection", self._logits([_semantic(item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind == "transfer_stack":
            exchange = self.offers.panel.armies
            choices = exchange.choices()
            sources = tuple(dict.fromkeys(source for source, _, _ in choices))
            source_index = choose("source_slot", self._logits([_army_slot(self.observation, ref, receiving=False) for ref in sources]),
                                  sources.index(action.selection) if action else None)
            selection = sources[source_index]
            destinations = tuple(destination for source, destination, _ in choices if source == selection)
            destination_index = choose("destination_slot", self._logits([
                _node("transfer_pair", _army_slot(self.observation, selection, receiving=False),
                      _army_slot(self.observation, ref, receiving=True)) for ref in destinations]),
                destinations.index(action.destination) if action else None)
            destination = destinations[destination_index]
            subject = self.offers.context.subject
            low, maximum = 1, exchange.maximum(selection, destination)
            high = maximum
            features = self._features(_node("transfer_quantity", _army_slot(self.observation, selection, receiving=False),
                                           _army_slot(self.observation, destination, receiving=True)))
            depth = 0
            while low < high:
                middle = (low + high) // 2
                numeric = self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31))
                logits = self.model.quantity(torch.cat((self.state, features, numeric)))
                logits = logits + self.model._float((log1p(middle - low), log1p(high - middle - 1)))
                selected = int(action.count > middle) if action else None
                branch = choose(f"count:{low}:{high}", logits, selected)
                if branch: low = middle + 1
                else: high = middle
                depth += 1
            count = low
        elif kind in ("construct", "back", "recruit"):
            subject = self.offers.context.subject
            if kind == "construct":
                selection = self.offers.panel.building.ref
            elif kind == "recruit":
                recruit = self.offers.panel.recruitment
                selection, destination = recruit.selected_offer_ref, recruit.destination.ref
                low, high, maximum = 1, recruit.slider_max, recruit.slider_max
                chosen = next(card for card in recruit.offers if card.ref == selection)
                features = self._features(_node("quantity", _semantic(chosen), _semantic(recruit.destination)))
                depth = 0
                while low < high:
                    middle = (low + high) // 2
                    numeric = self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31))
                    logits = self.model.quantity(torch.cat((self.state, features, numeric)))
                    # A zero learned logit yields uniform integer mass even when
                    # the two subtree sizes differ. This is support geometry.
                    logits = logits + self.model._float((log1p(middle - low), log1p(high - middle - 1)))
                    selected = int(action.count > middle) if action else None
                    branch = choose(f"count:{low}:{high}", logits, selected)
                    if branch: low = middle + 1
                    else: high = middle
                    depth += 1
                count = low
        selected_action = c.Action(kind, subject, x, y, schema=self.offers.schema,
                                   panel=panel, selection=selection, count=count, destination=destination)
        if not self.offers.permits(selected_action) or action is not None and selected_action != action:
            raise ValueError("Conditional action construction disagrees with support")
        return selected_action, ActionEvaluation(torch.stack([part.log_prob for part in trace]).sum(), tuple(trace))

    def sample(self, generator=None):
        return self._walk(generator=generator)[0]

    def evaluate(self, action):
        return self._walk(action=action)[1]

    def log_prob(self, action):
        return self.evaluate(action).log_prob

    def trace(self, action):
        return self.evaluate(action).conditionals
