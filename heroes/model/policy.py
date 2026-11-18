"""The policy: a recurrent actor-critic over one player's typed game history.

Every frame a seat receives becomes a semantic tree (what happened) plus, when
it carries one, the current observation (map grid and visible entities). Trees
and map are encoded and fed to one GRU, the only memory the policy has. At a
decision the GRU state scores the offered actions through an autoregressive
distribution: action kind first, then subject, target, quantity.

Opaque engine references are never encoded; they only route to the entity they
name in the current observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from math import copysign, log1p, sqrt
from operator import attrgetter
from sys import getsizeof

import torch
from torch import Tensor, nn

from ..game import contract as c
from ..game import guard as interaction_guard

MODEL_SCHEMA = "heroes-context-model-v6"  # reads the expanded adventure history (heroes-player-history-v8)
_ACTIONS = ("move_to", "open_hero", "open_town", "end_turn", "close", "open_panel",
            "open_building", "construct", "back", "select_recruit", "recruit", "transfer_stack", "select_hero", "hire_hero", "split_stack", "merge_stacks", "swap_stacks",
            "open_stack", "select_upgrade", "upgrade_stack", "swap_garrison", "choose_dialog")
# Frames that carry bookkeeping instead of anything the player observed. The model skips them.
_METADATA = frozenset(("issued", "superseded"))


@dataclass(frozen=True)
class ModelConfig:
    schema: str = MODEL_SCHEMA
    hidden_size: int = 64
    feature_size: int = 32
    max_map_side: int = 256

    def __post_init__(self):
        if self.schema != MODEL_SCHEMA:
            raise ValueError(f"Unsupported model schema {self.schema!r}; this code reads {MODEL_SCHEMA}")
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


def weights_sha256(model):
    """Digest of the model configuration and of every named tensor's exact bytes."""
    digest = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        flat = value.reshape(-1).view(torch.uint8)
        data = bytearray(flat.numel())
        if data:
            torch.frombuffer(data, dtype=torch.uint8).copy_(flat)
        header = json.dumps((name, str(value.dtype), tuple(value.shape)), separators=(",", ":")).encode()
        for part in (header, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


# -- reuse of encodings while acting --------------------------------------------
#
# Every frame is encoded with its own tree cache (a dict from tree to tensor).
# While the weights stay as they are and autograd is off, the model also keeps a
# memo across frames: leaves, internal nodes and whole observations. A hit is the
# tensor the same computation returned before, so acting is unchanged, only faster.

# Key of the per-frame cache entry that holds the memo; no tree equals it.
_LEAF_MEMO = object()
_LEAF_MEMO_LIMIT = 65536
_NODE_MEMO_LIMIT = 16384  # internal-node entries keep their (larger) trees alive
_OBSERVATION_MEMO_BYTES = 64 << 20  # observation entries hold map cells and tree caches, so bound their bytes
_OBSERVATION_TENSOR_BYTES = 512  # a CPU tensor beyond its data: object, implementation, storage (measured ~450)
# Key of the per-frame cache entry that is False once the cache may hold a tree
# equal to, but not identical with, a later one (see _encode_root).
_EXACT_CACHE = object()


def _leaf_key(kind, value):
    """Exact memo key, or None to bypass: 1, 1.0 and True stay distinct, as do 0.0 and -0.0."""
    if kind == "text":
        return value if type(value) is str else None
    if type(value) is float:
        return None if value != value else (float, value, copysign(1.0, value))
    return (type(value), value) if type(value) in (int, bool) else None


def _exact_tree(tree):
    """True when equality is identity for every subtree of this tree.

    Tree equality treats 0.0 and -0.0 as equal and NaN as unequal, and 1 equals
    1.0 and True. Trees whose numbers are floats other than -0.0 and NaN and
    whose texts and tags are strings have no such distinctions: equal subtrees
    are exactly identical and encode to identical tensors.
    """
    stack = [tree]
    while stack:
        item = stack.pop()
        if type(item) is not tuple:
            return False
        kind = item[0] if item else None
        if kind == "number":
            if len(item) != 2 or type(item[1]) is not float or item[1] != item[1]:
                return False
            if item[1] == 0.0 and copysign(1.0, item[1]) < 0:
                return False
        elif kind == "text":
            if len(item) != 2 or type(item[1]) is not str:
                return False
        elif kind in ("node", "ordered"):
            if len(item) != 3 or type(item[1]) is not str or type(item[2]) is not tuple:
                return False
            stack.extend(item[2])
        else:
            return False
    return True


# Tokens of _record_key; no int, str or None equals them.
_TRUE, _FALSE, _TUPLE = object(), object(), object()
_RECORD_FIELDS = {}


def _record_fields(kind):
    """A getter of every field of a contract record class as a tuple, or None for any other type."""
    getter = _RECORD_FIELDS.get(kind)
    if getter is None and kind not in _RECORD_FIELDS:
        if isinstance(kind, type) and is_dataclass(kind) and kind.__module__ == c.__name__:
            names = tuple(field.name for field in fields(kind))
            getter = attrgetter(*names) if len(names) > 1 else lambda item, names=names: tuple(getattr(item, name) for name in names)
        _RECORD_FIELDS[kind] = getter
    return getter


def _record_key(value):
    """An exact, injective key of a value of contract records, tuples, int, str, bool and None, or None.

    The key lists every record's class and field values, every tuple's length
    and elements, and each bool as its own token, so 1 and True stay distinct.
    Equal keys therefore mean values of identical types and contents, which
    every encoding reads identically. Anything else (floats, other types) has
    no key and is never memoized.
    """
    out, stack = [], [value]
    append, extend, pop = out.append, stack.extend, stack.pop
    while stack:
        item = pop()
        kind = type(item)
        if kind is int or kind is str or item is None:
            append(item)
        elif kind is tuple:
            append(_TUPLE); append(len(item)); extend(item)
        elif kind is bool:
            append(_TRUE if item else _FALSE)
        elif (kind is c.Tile and type(item.position) is tuple and len(item.position) == 3
              and all(type(v) is int for v in item.position)
              and type(item.terrain) is int and type(item.road) is int
              and type(item.blocked) is bool and type(item.visitable) is bool):
            # The same tokens as the generic LIFO traversal, without pushing
            # and popping each cell field separately for every observation.
            out.extend((kind, _TRUE if item.visitable else _FALSE,
                    _TRUE if item.blocked else _FALSE, item.road, item.terrain,
                    _TUPLE, 3, item.position[2], item.position[1], item.position[0]))
        else:
            getter = _record_fields(kind)
            if getter is None:
                return None
            append(kind); extend(getter(item))
    return tuple(out)


def _observation_entry_bytes(store, key, entry):
    """An upper estimate of the memory one observation memo entry adds to ``store``.

    It charges the record key with its strings and uncached integers, the entry
    and its cache dict, every tree object the cache keys keep alive, once each,
    and each tensor the store does not hold yet, with its data. The trees are
    walked because a child equal to an earlier key is a different object that
    only its parent keeps, with its whole subtree. A tree shared with an earlier
    entry is charged again; a charged tensor stays alive in the entries until
    they are cleared, which also forgets the charges.
    """
    charged, added = store["charged"], entry[3]
    size = getsizeof(key) + getsizeof(entry) + getsizeof(added) + sum(
        getsizeof(item) for item in key if type(item) is str or (type(item) is int and not -5 <= item <= 256))
    seen, stack = set(), [tree for tree in added if type(tree) is tuple]
    while stack:
        tree = stack.pop()
        if id(tree) not in seen:
            seen.add(id(tree))
            size += getsizeof(tree) + getsizeof(tree[1])
            if len(tree) > 2:
                size += getsizeof(tree[2])
                stack.extend(tree[2])
    for tensor in (*entry[:3], *added.values()):
        if isinstance(tensor, Tensor) and id(tensor) not in charged:
            charged.add(id(tensor))
            size += tensor.element_size() * tensor.numel() + _OBSERVATION_TENSOR_BYTES
    return size


def _has_army_operation(armies, kind):
    """bool(c.army_operation_choices(armies, kind)), stopping at the first choice."""
    return any(c.army_operation_maximum(armies, kind, source.ref, destination.ref) > 0
               for first in armies for source in first.slots for second in armies for destination in second.slots)


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
    if observation is not None and observation.window is not None and observation.window.capability_panel is not None:
        return next((_capability_choice(choice, observation) for choice in observation.window.capability_panel.choices
                     if choice.ref == ref), _node("unresolved_selection"))
    if observation is not None and observation.window is not None and observation.window.dialog_panel is not None:
        panel = observation.window.dialog_panel
        return next((_dialog_choice(panel, item) for item in panel.choices if item.ref == ref), _node("unresolved_selection"))
    if observation is not None and observation.window is not None and observation.window.stack_panel is not None:
        return next((_semantic(item) for item in observation.window.stack_panel.offers if item.ref == ref), _node("unresolved_selection"))
    if observation is None or observation.window is None or observation.window.town_panel is None:
        return _node("unresolved_selection")
    panel = observation.window.town_panel
    for entry in (*panel.controls, *panel.buildings,
                  *((panel.building,) if panel.building else ()),
                  *(panel.recruitment.offers if panel.recruitment else ()),
                  *(panel.tavern.offers if panel.tavern else ())):
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
                           if action.kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks", "open_stack") else _selection(observation, action.selection))))
    if action.destination is not None:
        if action.kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            children.append(_node("destination", _army_slot(observation, action.destination, receiving=True)))
        else:
            panel = observation.window.town_panel.recruitment
            if panel is None or panel.destination.ref != action.destination:
                raise ValueError("Action destination has no displayed semantic binding")
            children.append(_node("destination", _semantic(panel.destination)))
    return _node("action", *children)


def _army_slot(observation, ref, *, receiving):
    window = observation.window if observation else None
    if window is not None and window.armies is not None:
        army, slot = c.army_slot(window.armies, ref)
    else:
        panel = window.town_panel if window else None
        exchange = panel.armies if panel else None
        army = (exchange.destination if receiving else exchange.source) if exchange else None
        slot = next((slot for slot in army.slots if slot.ref == ref), None) if army else None
    if slot is None:
        raise ValueError("Army slot has no current displayed semantic binding")
    return _node("army_slot", _node("role", _text(army.role)), _node("owner_subject", _subject(observation, army.subject)),
                 _node("requires_nonempty", _semantic(army.requires_nonempty)), _semantic(slot))


def _dialog_choice(panel, choice):
    # The ordinal only routes to an already displayed component. Its magnitude
    # and the opaque choice/query references never become learned features.
    component = panel.components[choice.component_index] if choice.kind == "select_component" else None
    return _node("dialog_choice", _text(choice.kind), _semantic(component))


def _capability_choice(choice, observation):
    return _node("capability_choice", _semantic(choice),
                 _node("actor", _subject(observation, choice.subject)),
                 _node("recipient", _subject(observation, choice.destination))
                 if choice.destination is not None else _node("no_recipient"))


def _window_semantic(observation):
    """Bind expanded inventory ownership without learning opaque references."""
    window = observation.window
    ordinary = _semantic(window)
    if window is None or window.capability_panel is None:
        return ordinary
    panel = window.capability_panel
    items = [_node("owned_item", _semantic(item), _subject(observation, item.subject)) for item in panel.items]
    choices = [_capability_choice(choice, observation) for choice in panel.choices]
    return _node("capability_window", ordinary, _set("bound_items", items), _set("bound_choices", choices),
                 _node("inspected_object", _subject(observation, panel.inspected_subject)))


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
    if type(value) is c.TavernPanel:
        cards = [_node("card", _semantic(card), _semantic(card.ref == value.selected_offer_ref)) for card in value.offers]
        return _node("tavern", _set("cards", cards), _node("selected_detail", _semantic(value.selected_detail)),
                     _node("gold_cost", _number(value.gold_cost)), _node("recruit_status", _text(value.recruit_status)))
    if type(value) is c.StackPanel:
        cards = [_node("upgrade", _semantic(card), _semantic(card.ref == value.selected_upgrade_ref)) for card in value.offers]
        return _node("stack_panel", _semantic(value.parent_context), _semantic(value.source), _set("offers", cards),
                     _node("total_cost", _semantic(value.total_cost)), _node("affordable", _semantic(value.affordable)))
    if type(value) is c.DialogPanel:
        return _node("dialog_panel", _text(value.family), _semantic(value.text),
                     _set("components", [_semantic(item) for item in value.components]),
                     _node("selection", _semantic(value.selection)), _node("cancel", _semantic(value.cancel)),
                     _set("choices", [_dialog_choice(value, choice) for choice in value.choices]))
    if type(value) is c.DialogChoice:
        raise ValueError("Dialog choice requires its displayed component binding")
    if type(value) is c.Notice:
        return _node("notice", _text(value.display_mode),
                     ("ordered", "notice_text", tuple(_semantic(token) for token in value.text)),
                     _set("components", [_semantic(component) for component in value.components]))
    if is_dataclass(value):
        children = []
        for name in c._field_names(type(value)):
            item = getattr(value, name)
            # Additive profile fields must not add even an "absent" node to
            # old observations: archived values/probabilities remain exact.
            if item is None and (type(value) is c.TownPanel and name in ("armies", "tavern")
                                 or type(value) is c.Hero and name == "in_garrison"
                                 or type(value) is c.Window and name in ("armies", "stack_panel", "dialog_panel", "capability_panel", "capability_controls")
                                 or type(value) is c.VisibleEvent and name in ("partner", "dialog", "capability")
                                 or type(value) is c.NoticeComponent and name == "skill"):
                continue
            if type(value) is c.TownPanel and name in ("armies", "tavern") and item is None:
                continue
            if name in ("handle", "ref", "control_ref", "selected_offer_ref", "selected_upgrade_ref", "subject", "selection", "partner", "schema", "inspected_subject"):
                continue
            if type(value) is c.CapabilityChoice and name == "destination":
                continue  # requires its displayed recipient binding, never handle bytes
            if (name in ("creature", "terrain", "road", "kind", "subtype", "object_kind", "object_subtype", "faction", "portrait", "hero_class")
                    or type(value) in (c.CapabilityChoice, c.CapabilityItem, c.CapabilityPanel)
                    and name in ("id", "artifact", "spell", "skill", "slot", "destination_slot", "resource_from", "resource_to", "primary_skill")) and type(item) is int:
                encoded = _text(f"category:{item}")
            elif type(value) is c.CapabilityItem and value.kind == "world_object" and name == "amount" and type(item) is int:
                encoded = _text(f"category:{item}")  # delivered object subtype, not an amount
            elif name in ("children", "controls", "buildings", "army", "items", "choices", "capability_controls") and type(item) is tuple:
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
    # The frame's own observation, never an accumulated map.
    observation: c.Observation | None = None
    guard: interaction_guard.GuardSupport | None = None


class ContextualEncoder:
    """Turns one seat's frames, in order, into model inputs. It holds no neural state.

    It keeps only what resolves references: the latest observation, and the
    observation of the decision that is still open, which is the one the
    submitted action's references were chosen from.
    """

    def __init__(self, guard=None):
        self.observation = None
        self.decision = None  # the decision frame whose action has not finished yet
        self.guard_tracker = interaction_guard.Tracker(guard)

    def encode(self, frame):
        basis = self.decision.observation if self.decision is not None else self.observation
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
                if frame.event.partner is not None:
                    children.append(_node("prior_observation_partner", _subject(self.observation, frame.event.partner)))
            if frame.result is not None:
                children.append(_semantic(frame.result))
            if frame.context is not None:
                children.append(_semantic(frame.context))
                children.append(_node("prior_observation_subject", _subject(self.observation, frame.context.subject)))
                if frame.context.kind == "exchange":
                    children.append(_node("prior_observation_partner", _subject(self.observation, frame.context.selection)))
            if frame.accepted is not None:
                children.append(_node("accepted", _semantic(frame.accepted)))
            if frame.outcome is not None:
                children.append(_node("observed_outcome", _text(frame.outcome)))
            # Consumer layer, revisions, sequence numbers, evidence and command
            # counters are deliberately absent. A consumed-action marker remains.
            payload = _node("frame", *children)
        guard = self.guard_tracker.observe(frame)
        if kind == "decision":
            self.decision = frame
        elif kind in ("result", "superseded"):
            self.decision = None
        if frame.observation is not None:
            self.observation = frame.observation
        return ContextualInput(frame, payload, frame.observation, guard)


def encode_history(frames, *, guard=None):
    """The model inputs of one seat's complete frame sequence."""
    encoder = ContextualEncoder(guard)
    return tuple(encoder.encode(frame) for frame in frames)


@dataclass(frozen=True)
class ConditionalProbability:
    name: str
    selected_index: int
    support_size: int
    log_prob: Tensor


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


def observation_semantics(observation):
    """The visible semantic tree shared by Python and native tensor execution."""
    own = _node("observation", _node("size", _semantic(observation.size)),
                _node("resources", _semantic(observation.resources)),
                _set("heroes", [_semantic(hero) for hero in observation.heroes]),
                _set("towns", [_semantic(town) for town in observation.towns]),
                _window_semantic(observation),
                _node("current_window_subject", _subject(observation, observation.context.subject)))
    if observation.window is not None and observation.window.armies is not None:
        window = observation.window
        bindings = [_node("displayed_army", _semantic(army), _subject(observation, army.subject)) for army in window.armies]
        if window.stack_panel is not None:
            parent = window.stack_panel.parent_context
            bindings.append(_node("stack_parent", _subject(observation, parent.subject),
                                  _subject(observation, parent.selection) if parent.kind == "exchange" else _node("absent")))
        if window.context.kind == "exchange":
            bindings.append(_node("exchange_partner", _subject(observation, window.context.selection)))
        own = _node("management_observation", own, _set("army_bindings", bindings))
    if observation.own_armies is not None:
        bindings = [_node("own_army", _semantic(army), _subject(observation, army.subject))
                    for army in observation.own_armies]
        own = _node("passive_army_observation", own, _set("own_army_bindings", bindings))
    return own


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

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_memo_state", None)  # copies and pickles start without a memo
        return state

    def _memo(self):
        """The encodings reused across frames, or None when nothing may be reused.

        Reuse needs evaluation mode without autograd: a hit is then exactly the
        tensor a recomputation would return. The memo belongs to one version of
        the weights and starts empty as soon as any parameter was written in
        place (an optimizer step, ``copy_``), given new storage (``.to``) or
        loaded. A write through ``parameter.data`` is the one change this
        cannot see; nothing in the package does that.
        """
        if torch.is_grad_enabled() or self.training:
            return None
        memo = self.__dict__.get("_memo_state")
        parameters = tuple(self.parameters()) if memo is None else memo["parameters"]
        versions = tuple((parameter._version, parameter.data_ptr()) for parameter in parameters)
        if memo is None or memo["versions"] != versions:
            memo = self.__dict__["_memo_state"] = {
                "parameters": parameters, "versions": versions, "text": {}, "number": {}, "node": {},
                "observations": {"entries": {}, "bytes": 0, "charged": set()}}
        return memo

    def load_state_dict(self, *arguments, **options):
        self.__dict__.pop("_memo_state", None)  # ``assign=True`` replaces the parameter objects themselves
        return super().load_state_dict(*arguments, **options)

    def _encode_root(self, tree, cache, final=False):
        """Encode one top-level tree with this frame's cache, as ``_encode_tree`` does.

        Internal nodes come from the inference memo only while this tree and
        every earlier root of the cache are exact. Then no lookup can match an
        equal but different tree, so every subtree's cached value is the exact
        encoding of that very subtree, the value a memo entry holds. Hits still
        visit their subtrees, so the cache gains exactly the entries the
        reference computation would add, and a later inexact root (signed zero,
        NaN) meets the reference cache. Once a root is inexact, or a root was
        encoded without the memo, the cache no longer uses it.
        """
        memo = cache.get(_LEAF_MEMO)
        nodes = memo["node"] if memo is not None and not torch.is_grad_enabled() else None
        if nodes is not None and cache.get(_EXACT_CACHE, True) and _exact_tree(tree):
            cache[_EXACT_CACHE] = True
            # The last root of a cache that nothing reads afterwards needs no visit.
            hit = nodes.get(tree) if final else None
            return hit if hit is not None else self._encode_tree(tree, cache, nodes)
        cache[_EXACT_CACHE] = False
        return self._encode_tree(tree, cache)

    def _encode_tree(self, tree, cache, nodes=None):
        if tree in cache:
            return cache[tree]
        kind = tree[0]
        memo = cache.get(_LEAF_MEMO) if kind in ("text", "number") else None
        key = None if memo is None or torch.is_grad_enabled() else _leaf_key(kind, tree[1])
        if key is not None:
            entries = memo[kind]
            result = entries.get(key)
            if result is None:
                result = self._encode_tree(tree, {})
                if len(entries) >= _LEAF_MEMO_LIMIT:
                    entries.clear()  # bounded memory; a miss only recomputes
                entries[key] = result
            cache[tree] = result
            return result
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
            result = nodes.get(tree) if nodes is not None else None
            if result is not None:
                # Visit the subtrees as the reference does, adding exactly its cache entries.
                for child in children:
                    self._encode_tree(child, cache, nodes)
                self._encode_tree(_text(tag), cache, nodes)
                cache[tree] = result
                return result
            encodings = [self._encode_tree(child, cache, nodes) for child in children]
            if not encodings:
                pooled = self.byte.weight.new_zeros(self.config.feature_size)
            elif kind == "ordered":
                pooled = self.ordered(torch.stack(encodings).unsqueeze(0))[1][0, 0]
            else:
                pooled = torch.stack(encodings).sum(0) / sqrt(len(encodings))
            result = self.tree(torch.cat((self._encode_tree(_text(tag), cache, nodes), pooled)))
            if nodes is not None:
                if len(nodes) >= _NODE_MEMO_LIMIT:
                    nodes.clear()  # bounded memory; a miss only recomputes
                nodes[tree] = result
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
        return global_map, self._encode_root(observation_semantics(observation), cache), cells[0].permute(1, 2, 0).reshape(-1, 16)

    def _check_step(self, frame, state):
        if type(frame) is not ContextualInput:
            raise ValueError("Encode frames with ContextualEncoder first")
        if not c.profile_for(frame.frame.schema).capabilities:
            raise ValueError("This model reads the expanded adventure history only")
        if (not isinstance(state, Tensor) or state.shape != (self.config.hidden_size,)
                or state.device != self.device or state.dtype != self.byte.weight.dtype):
            raise ValueError("State must match model dimensions, dtype and device")

    def _frame_features(self, frame, cache, final=False):
        """State-independent features; callers retain their own per-frame tree cache.

        ``final`` means nothing reads the cache after these features.
        """
        zero = self.byte.weight.new_zeros(self.config.feature_size)
        if frame.observation is None:
            map_features = own = zero
            cells = None
        else:
            map_features, own, cells = self._observation_features(frame.observation, cache)
        return torch.cat((self._encode_root(frame.semantics, cache, final), map_features, own)), cells

    def _observation_features(self, observation, cache):
        """``_encode_observation``, reused for an observation with identical contents.

        Only with the memo and a cache that holds nothing else yet, so the
        encoding and the tree cache entries it adds depend on the observation
        alone (exactly keyed by ``_record_key``). A hit returns the stored
        tensors and adds exactly the stored cache entries, as the computation would.
        """
        memo = cache.get(_LEAF_MEMO)
        store = memo["observations"] if memo is not None and len(cache) == 1 and not torch.is_grad_enabled() else None
        key = _record_key(observation) if store is not None else None
        if key is None:
            return self._encode_observation(observation, cache)
        entries = store["entries"]
        hit = entries.get(key)
        if hit is not None:
            global_map, own, cells, added = hit
            cache.update(added)
            return global_map, own, cells
        global_map, own, cells = self._encode_observation(observation, cache)
        entry = (global_map, own, cells, {k: v for k, v in cache.items() if k is not _LEAF_MEMO})
        size = _observation_entry_bytes(store, key, entry)
        if store["bytes"] + size > _OBSERVATION_MEMO_BYTES:
            entries.clear()  # bounded memory; a miss only recomputes
            store["bytes"], store["charged"] = 0, set()
            size = _observation_entry_bytes(store, key, entry)
        store["bytes"] += size
        entries[key] = entry
        return global_map, own, cells

    def forward_step(self, frame, state):
        self._check_step(frame, state)
        if frame.semantics is None:
            return StepOutput(state, None)
        cache = {}
        memo = self._memo()
        if memo is not None:
            cache[_LEAF_MEMO] = memo
        vector, cells = self._frame_features(frame, cache, final=frame.frame.kind != "decision")
        state = self.gru(vector, state)
        decision = None
        if frame.frame.kind == "decision":
            distribution = ContextualDistribution(self, state, frame.frame.observation, frame.frame.offers, cells, cache, frame.guard)
            decision = PolicyValue(self.value(state).squeeze(-1), distribution)
        return StepOutput(state, decision)


class ContextualDistribution:
    """Typed autoregressive distribution, exact support and chosen-path log P.

    Quantity uses a balanced binary interval tree. At most 31 learned choices
    represent every legal int32 count. There is deliberately no entropy method:
    exact joint count entropy would require visiting every leaf. PPO starts with
    entropy coefficient zero, not a selected-path value mislabeled as entropy.
    """

    def __init__(self, model, state, observation, offers, cells, cache, guard=None):
        self.model, self.state, self.observation, self.offers = model, state, observation, offers
        self.cells, self.cache = cells, cache
        self.guard = guard

    def _features(self, semantic):
        return self.model._encode_root(semantic, self.cache)

    def _logits(self, semantics):
        candidates = torch.stack([self._features(item) for item in semantics])
        return candidates.mv(torch.tanh(self.model.policy_query(self.state))) / sqrt(self.model.config.feature_size)

    def _quantity_logits(self, features, low, high, maximum, depth):
        numeric = self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31))
        logits = self.model.quantity(torch.cat((self.state, features, numeric)))
        middle = (low + high) // 2
        return logits + self.model._float((log1p(middle - low), log1p(high - middle - 1)))

    def _coordinate_positions(self, subject, width, height):
        hx = hy = 0
        if self.guard is not None:  # guarded policies see coordinates relative to the moving hero
            hx, hy, _ = next(hero.position for hero in self.observation.heroes if hero.handle == subject)
        return self.model._float([((x - hx) / max(width - 1, 1), (y - hy) / max(height - 1, 1))
                                  for y in range(height) for x in range(width)])

    def _coordinate_logits(self, semantic, subject, width, height):
        positions = self._coordinate_positions(subject, width, height)
        target = self._features(semantic).expand(width * height, -1)
        state = self.state.expand(width * height, -1)
        return self.model.coordinate(torch.cat((state, target, self.cells, positions), -1)).squeeze(-1)

    def _coordinate_subset(self, logits, coordinates):
        return logits[torch.tensor(coordinates, dtype=torch.long, device=logits.device)]

    def _choose(self, name, logits, selected, generator, trace):
        log_probs = torch.log_softmax(logits, 0)
        if selected is None:
            selected = int(torch.multinomial(log_probs.detach().exp(), 1, generator=generator).item())
        trace.append(ConditionalProbability(name, selected, logits.numel(), log_probs[selected]))
        return selected

    def _kinds(self):
        kinds = self._native_kinds()
        if self.guard is None:
            return kinds
        return tuple(kind for kind in kinds if any(self._permitted(action) for action in self._representatives(kind)))

    def _native_kinds(self):
        if self.offers.capability_panel is not None:
            panel = self.offers.capability_panel
            return (("close_capability",) if not panel.mandatory else ()) + tuple(
                kind for kind in c.CAPABILITY_ACTIONS if any(choice.kind == kind for choice in panel.choices))
        kinds = self._window_kinds()
        return kinds + (("open_capability",) if self.offers.capability_controls else ())

    def _window_kinds(self):
        offers, context = self.offers, self.offers.context
        if context.kind == "dialog":
            return ("choose_dialog",)
        if context.kind == "map":
            movable = offers.hero_refs if offers.movable_hero_refs is None else offers.movable_hero_refs
            return tuple(kind for kind in _ACTIONS if kind == "end_turn"
                         or kind == "move_to" and movable or kind == "open_hero" and offers.hero_refs
                         or kind == "open_town" and offers.town_refs)
        if context.kind == "stack":
            if context.panel == "overview":
                return ("back",) + (("select_upgrade",) if offers.stack_panel.offers else ())
            return ("back",) + (("upgrade_stack",) if offers.stack_panel.affordable else ())
        if offers.armies:
            kinds = ("close",) + (("open_panel",) if offers.panel and offers.panel.controls else ())
            kinds += tuple(kind for kind in ("split_stack", "merge_stacks", "swap_stacks") if _has_army_operation(offers.armies, kind))
            kinds += (("open_stack",) if any(slot.count for army in offers.armies for slot in army.slots) else ())
            if context.kind == "town" and any(army.role in ("garrison_hero", "visiting_hero") for army in offers.armies):
                kinds += ("swap_garrison",)
            return kinds
        if offers.panel is None:
            return ("close",)
        if context.panel == "overview":
            kinds = ("close", "open_panel") if offers.panel.controls else ("close",)
            return kinds + (("transfer_stack",) if offers.panel.armies and offers.panel.armies.choices() else ())
        if context.panel == "hall":
            return ("open_building", "back") if offers.panel.buildings else ("back",)
        if context.panel == "building":
            return ("construct", "back") if offers.panel.building.status == "allowed" else ("back",)
        if context.panel == "tavern":
            tavern = offers.panel.tavern
            return (("back",) + (("select_hero",) if tavern.offers else ())
                    + (("hire_hero",) if tavern.recruit_status == "allowed" else ()))
        recruitment = offers.panel.recruitment
        return ("back", "select_recruit", "recruit") if recruitment.slider_max else ("back", "select_recruit")

    def _action(self, kind, **values):
        subject = None if kind in ("close", "end_turn") else self.offers.context.subject
        return c.Action(kind, schema=self.offers.schema, **dict({"subject": subject}, **values))

    def _permitted(self, action):
        return self.guard is None or self.guard.permits(action, self.observation, self.offers)

    def _filtered(self, choices, action_for):
        # Quantity-bearing choices supply a permitted representative; their
        # count tree separately excludes an immediately repeated quantity.
        if self.guard is None:
            return choices
        return tuple(choice for choice in choices if self._permitted(action_for(choice)))

    def _coordinates(self, subject):
        return self.guard.movement_coordinates(subject, self.observation, self.offers)

    def _subjects(self, kind):
        refs = (self.offers.town_refs if kind == "open_town" else self.offers.movable_hero_refs
                if kind == "move_to" and self.offers.movable_hero_refs is not None else self.offers.hero_refs)
        if self.guard is None:
            return refs
        if kind == "move_to":
            return tuple(ref for ref in refs if ref in self.guard.movable_heroes
                         and next(self._coordinates(ref), None) is not None)
        return self._filtered(refs, lambda ref: self._action(kind, subject=ref))

    def _panel_action(self, control):
        return self._action("open_panel", panel=control.panel,
                            selection=control.ref if control.panel == "recruitment" else None)

    def _pair_action(self, kind, pair):
        source, destination, maximum = pair
        if kind in ("transfer_stack", "split_stack"):
            return self._quantity_representative(kind, maximum, selection=source, destination=destination)
        return self._action(kind, selection=source, destination=destination)

    def _repeated_quantity(self, action):
        return self.guard.repeated_quantity(action, self.observation, self.offers) if self.guard else None

    def _quantity_representative(self, kind, maximum, **values):
        action = self._action(kind, count=1, **values)
        if maximum > 1 and self._repeated_quantity(action) == 1:
            return self._action(kind, count=2, **values)
        return action

    def _choose_count(self, choose, features, maximum, selected, excluded):
        low, high, depth = 1, maximum, 0
        while low < high:
            middle = (low + high) // 2
            # If a branch contains only the excluded quantity, take its sibling.
            if excluded == low == middle:
                branch = 1
            elif excluded == high == middle + 1:
                branch = 0
            else:
                logits = self._quantity_logits(features, low, high, maximum, depth)
                branch = choose(f"count:{low}:{high}", logits, int(selected > middle) if selected else None)
            if branch:
                low = middle + 1
            else:
                high = middle
            depth += 1
        return low

    def _representatives(self, kind):
        """One completion per non-quantity leaf, used before choosing a kind.

        Movement stays lazy here, so testing whether a hero has a destination
        never materializes the rectangle or scores a second policy distribution.
        """
        offers = self.offers
        if kind in ("move_to", "open_hero", "open_town"):
            for ref in self._subjects(kind):
                if kind == "move_to":
                    index = next(self._coordinates(ref), None)
                    if index is not None:
                        yield self._action(kind, subject=ref, x=index % offers.size[0], y=index // offers.size[0])
                else:
                    yield self._action(kind, subject=ref)
        elif kind == "open_capability":
            for panel in offers.capability_controls:
                yield self._action(kind, panel=panel)
        elif kind in c.CAPABILITY_ACTIONS:
            for choice in offers.capability_panel.choices:
                if choice.kind == kind:
                    yield (self._quantity_representative(kind, choice.max_count, selection=choice.ref)
                           if kind == "trade_resource" else self._action(kind, selection=choice.ref))
        elif kind == "choose_dialog":
            for choice in offers.dialog_panel.choices:
                yield self._action(kind, selection=choice.ref)
        elif kind == "open_panel":
            for control in offers.panel.controls:
                yield self._panel_action(control)
        elif kind == "open_stack":
            for army in offers.armies:
                for slot in army.slots:
                    if slot.count:
                        yield self._action(kind, selection=slot.ref)
        elif kind in ("select_upgrade", "open_building", "select_recruit", "select_hero"):
            choices = (offers.stack_panel.offers if kind == "select_upgrade" else
                       offers.panel.buildings if kind == "open_building" else
                       offers.panel.tavern.offers if kind == "select_hero" else offers.panel.recruitment.offers)
            for choice in choices:
                yield self._action(kind, selection=choice.ref)
        elif kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            choices = (offers.panel.armies.choices() if kind == "transfer_stack"
                       else c.army_operation_choices(offers.armies, kind))
            for pair in choices:
                yield self._pair_action(kind, pair)
        elif kind == "recruit":
            recruit = offers.panel.recruitment
            yield self._quantity_representative(kind, recruit.slider_max, selection=recruit.selected_offer_ref,
                                                destination=recruit.destination.ref)
        elif kind == "construct":
            yield self._action(kind, selection=offers.panel.building.ref)
        elif kind == "hire_hero":
            yield self._action(kind, selection=offers.panel.tavern.selected_offer_ref)
        elif kind == "upgrade_stack":
            yield self._action(kind, selection=offers.stack_panel.selected_upgrade_ref)
        else:
            yield self._action(kind)

    def _walk(self, action=None, generator=None):
        if action is not None and (not self.offers.permits(action) or not self._permitted(action)):
            raise ValueError("Action is outside this exact behavior support")
        memo = self.model._memo()  # revalidated: weights or modes may differ from forward_step
        if memo is not self.cache.get(_LEAF_MEMO):
            # Values cached under other weights or modes must never seed internal-node entries.
            self.cache[_EXACT_CACHE] = False
        if memo is None:
            self.cache.pop(_LEAF_MEMO, None)
        else:
            self.cache[_LEAF_MEMO] = memo
        trace = []

        def choose(name, logits, selected=None):
            return self._choose(name, logits, selected, generator, trace)

        kinds = self._kinds()
        kind = kinds[choose("kind", self._logits([_node("action_kind", _text(kind)) for kind in kinds]),
                            kinds.index(action.kind) if action else None)]
        subject = panel = selection = destination = x = y = count = None
        if kind == "open_capability":
            controls = self._filtered(self.offers.capability_controls, lambda panel: self._action(kind, panel=panel))
            index = choose("capability_panel", self._logits([_node("capability_panel", _text(item)) for item in controls]),
                           controls.index(action.panel) if action else None)
            subject, panel = self.offers.context.subject, controls[index]
        elif kind == "close_capability":
            subject = self.offers.context.subject
        elif kind in c.CAPABILITY_ACTIONS:
            choices = self._filtered(tuple(item for item in self.offers.capability_panel.choices if item.kind == kind),
                                     lambda item: self._quantity_representative(kind, item.max_count, selection=item.ref)
                                     if kind == "trade_resource" else self._action(kind, selection=item.ref))
            index = choose("capability_choice", self._logits([_capability_choice(item, self.observation) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            choice = choices[index]
            subject, selection = self.offers.context.subject, choice.ref
            if kind == "trade_resource":
                features = self._features(_capability_choice(choice, self.observation))
                excluded = self._repeated_quantity(self._action(kind, selection=selection, count=1))
                count = self._choose_count(choose, features, choice.max_count, action.count if action else None, excluded)
        elif kind == "choose_dialog":
            delivered = self.offers.dialog_panel
            choices = delivered.choices
            index = choose("dialog_choice", self._logits([_dialog_choice(delivered, item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("move_to", "open_hero", "open_town"):
            refs = self._subjects(kind)
            semantics = [_node(kind, _subject(self.observation, ref)) for ref in refs]
            index = choose("subject", self._logits(semantics), refs.index(action.subject) if action else None)
            subject = refs[index]
            if kind == "move_to":
                width, height, _ = self.offers.size
                logits = self._coordinate_logits(semantics[index], subject, width, height)
                if self.guard is None:
                    index = choose("coordinate", logits, action.y * width + action.x if action else None)
                else:
                    coordinates = tuple(self._coordinates(subject))
                    logits = self._coordinate_subset(logits, coordinates)
                    index = coordinates[choose("coordinate", logits,
                                        coordinates.index(action.y * width + action.x) if action else None)]
                x, y = index % width, index // width
        elif kind == "open_panel":
            controls = self.offers.panel.controls
            # Hall and tavern each have one semantic navigation action;
            # duplicate controls must never double-count its probability mass.
            if any(sum(control.panel == panel for control in controls) > 1 for panel in ("hall", "tavern")):
                raise ValueError("Duplicate semantic town panel controls")
            controls = self._filtered(controls, self._panel_action)
            selected = next((i for i, control in enumerate(controls) if action and control.panel == action.panel
                             and (control.panel in ("hall", "tavern") or control.ref == action.selection)), None)
            index = choose("control", self._logits([_semantic(control) for control in controls]), selected)
            control = controls[index]
            subject, panel = self.offers.context.subject, control.panel
            selection = control.ref if panel == "recruitment" else None
        elif kind == "open_stack":
            choices = tuple(slot.ref for army in self.offers.armies for slot in army.slots if slot.count)
            choices = self._filtered(choices, lambda ref: self._action(kind, selection=ref))
            index = choose("selection", self._logits([_army_slot(self.observation, ref, receiving=False) for ref in choices]),
                           choices.index(action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index]
        elif kind == "select_upgrade":
            choices = self._filtered(self.offers.stack_panel.offers, lambda item: self._action(kind, selection=item.ref))
            index = choose("selection", self._logits([_semantic(item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("open_building", "select_recruit", "select_hero"):
            choices = (self.offers.panel.buildings if kind == "open_building" else
                       self.offers.panel.tavern.offers if kind == "select_hero" else self.offers.panel.recruitment.offers)
            choices = self._filtered(choices, lambda item: self._action(kind, selection=item.ref))
            index = choose("selection", self._logits([_semantic(item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            exchange = self.offers.panel.armies if kind == "transfer_stack" else None
            choices = exchange.choices() if exchange else c.army_operation_choices(self.offers.armies, kind)
            choices = self._filtered(choices, lambda pair: self._pair_action(kind, pair))
            sources = tuple(dict.fromkeys(source for source, _, _ in choices))
            source_semantics = [_army_slot(self.observation, ref, receiving=False) for ref in sources]
            if kind != "transfer_stack": source_semantics = [_node(kind, item) for item in source_semantics]
            source_index = choose("source_slot", self._logits(source_semantics),
                                  sources.index(action.selection) if action else None)
            selection = sources[source_index]
            destinations = tuple(destination for source, destination, _ in choices if source == selection)
            destination_index = choose("destination_slot", self._logits([
                _node("transfer_pair" if kind == "transfer_stack" else kind + "_pair", _army_slot(self.observation, selection, receiving=False),
                      _army_slot(self.observation, ref, receiving=True)) for ref in destinations]),
                destinations.index(action.destination) if action else None)
            destination = destinations[destination_index]
            subject = self.offers.context.subject
            maximum = (exchange.maximum(selection, destination) if exchange else
                               c.army_operation_maximum(self.offers.armies, kind, selection, destination))
            features = self._features(_node("transfer_quantity" if kind == "transfer_stack" else kind + "_quantity", _army_slot(self.observation, selection, receiving=False),
                                           _army_slot(self.observation, destination, receiving=True)))
            if kind in ("transfer_stack", "split_stack"):
                excluded = self._repeated_quantity(self._action(kind, selection=selection, destination=destination, count=1))
                count = self._choose_count(choose, features, maximum, action.count if action else None, excluded)
        elif kind in ("construct", "back", "recruit", "hire_hero", "swap_garrison", "upgrade_stack"):
            subject = self.offers.context.subject
            if kind == "upgrade_stack":
                selection = self.offers.stack_panel.selected_upgrade_ref
            elif kind == "construct":
                selection = self.offers.panel.building.ref
            elif kind == "hire_hero":
                selection = self.offers.panel.tavern.selected_offer_ref
            elif kind == "recruit":
                recruit = self.offers.panel.recruitment
                selection, destination = recruit.selected_offer_ref, recruit.destination.ref
                chosen = next(card for card in recruit.offers if card.ref == selection)
                features = self._features(_node("quantity", _semantic(chosen), _semantic(recruit.destination)))
                excluded = self._repeated_quantity(self._action(kind, selection=selection, destination=destination, count=1))
                count = self._choose_count(choose, features, recruit.slider_max, action.count if action else None, excluded)
        selected_action = c.Action(kind, subject, x, y, schema=self.offers.schema,
                                   panel=panel, selection=selection, count=count, destination=destination)
        if not self.offers.permits(selected_action) or not self._permitted(selected_action) or action is not None and selected_action != action:
            raise ValueError("Conditional action construction disagrees with support")
        return selected_action, ActionEvaluation(torch.stack([part.log_prob for part in trace]).sum(), tuple(trace))

    def sample_evaluated(self, generator=None):
        """A sampled action with its evaluation: only the selected indices come from the generator."""
        return self._walk(generator=generator)

    def evaluate(self, action):
        return self._walk(action=action)[1]

    def log_prob(self, action):
        return self.evaluate(action).log_prob
