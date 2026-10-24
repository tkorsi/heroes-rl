"""Strict contextual history contract; no engine adapter or learning updates.

All observations must already have been filtered by an audited player callback.
Validation checks shape and chronology, not whether a producer told the truth
about visibility. Routing handles, revisions and evidence are bookkeeping, never
numeric neural features. Frozen records preserve every delivered prefix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import re

SCHEMA = "heroes-player-history-v2"
ACTION_SCHEMA = "heroes-context-actions-v2"
ENVIRONMENT = "adventure-windows-v1"
TOWN_SCHEMA = "heroes-player-history-v3"
TOWN_ACTION_SCHEMA = "heroes-context-actions-v3"
TOWN_ENVIRONMENT = "adventure-town-v1"
ARMIES_SCHEMA = "heroes-player-history-v4"
ARMIES_ACTION_SCHEMA = "heroes-context-actions-v4"
ARMIES_ENVIRONMENT = "adventure-armies-v1"
HEROES_SCHEMA = "heroes-player-history-v5"
HEROES_ACTION_SCHEMA = "heroes-context-actions-v5"
HEROES_ENVIRONMENT = "adventure-heroes-v1"
TRAJECTORY_SCHEMA = "heroes-context-trajectory-v1"
TERMINAL_TRAJECTORY_SCHEMA = "heroes-context-trajectory-v2"
MAX_INT = 2**63 - 1


@dataclass(frozen=True)
class InterfaceProfile:
    environment: str
    history_schema: str
    action_schema: str
    protocol: int
    version: int

    @property
    def history_key(self):
        return f"history_v{self.version}"

    @property
    def history_filename(self):
        return f"history-v{self.version}.jsonl"

    @property
    def town_economy(self):
        return self.version in (3, 4, 5)

    @property
    def army_transfer(self):
        return self.version in (4, 5)

    @property
    def hero_recruitment(self):
        return self.version == 5


WINDOWS_PROFILE = InterfaceProfile(ENVIRONMENT, SCHEMA, ACTION_SCHEMA, 4, 2)
TOWN_PROFILE = InterfaceProfile(TOWN_ENVIRONMENT, TOWN_SCHEMA, TOWN_ACTION_SCHEMA, 5, 3)
ARMIES_PROFILE = InterfaceProfile(ARMIES_ENVIRONMENT, ARMIES_SCHEMA, ARMIES_ACTION_SCHEMA, 6, 4)
HEROES_PROFILE = InterfaceProfile(HEROES_ENVIRONMENT, HEROES_SCHEMA, HEROES_ACTION_SCHEMA, 7, 5)


def profile_for(value):
    """Select an explicit interface/history/action version; v2 remains the default."""
    for profile in (WINDOWS_PROFILE, TOWN_PROFILE, ARMIES_PROFILE, HEROES_PROFILE):
        if value in (profile.environment, profile.history_schema, profile.action_schema):
            return profile
    raise ValueError("Unsupported contextual interface/schema")


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
    panel: str | None = None
    selection: str | None = None

    def __post_init__(self):
        if self.kind not in ("map", "hero", "town"):
            raise ValueError("Unsupported context")
        if self.kind == "map":
            if self.subject is not None:
                raise ValueError("Map context has no subject")
        else:
            _text(self.subject)
        if self.kind != "town" and (self.panel is not None or self.selection is not None):
            raise ValueError("Only town contexts have panels")
        if self.panel not in (None, "overview", "hall", "building", "recruitment", "tavern"):
            raise ValueError("Unsupported town panel")
        if self.panel in ("building", "recruitment"):
            _text(self.selection)
        elif self.selection is not None:
            raise ValueError("Only building or recruitment panels have a selection")


@dataclass(frozen=True)
class Action:
    kind: str
    subject: str | None = None
    x: int | None = None
    y: int | None = None
    schema: str = ACTION_SCHEMA
    panel: str | None = None
    selection: str | None = None
    count: int | None = None
    destination: str | None = None

    def __post_init__(self):
        profile = profile_for(self.schema)
        if self.schema != profile.action_schema:
            raise ValueError("Expected an action schema")
        town_actions = ("open_panel", "open_building", "construct", "back", "select_recruit", "recruit")
        if profile.army_transfer:
            town_actions += ("transfer_stack",)
        if profile.hero_recruitment:
            town_actions += ("select_hero", "hire_hero")
        if self.kind not in ("move_to", "open_hero", "open_town", "close", "end_turn", *(
                town_actions if profile.town_economy else ())):
            raise ValueError("Unsupported contextual action")
        if self.kind in ("move_to", "open_hero", "open_town", *town_actions):
            _text(self.subject)
        elif self.subject is not None:
            raise ValueError("This action has no subject")
        if self.kind == "move_to":
            _integer(self.x)
            _integer(self.y)
        elif self.x is not None or self.y is not None:
            raise ValueError("Only destination movement takes coordinates")
        if self.kind == "open_panel":
            if self.panel == "recruitment":
                _text(self.selection)
            elif self.panel not in (("hall", "tavern") if profile.hero_recruitment else ("hall",)) or self.selection is not None:
                raise ValueError("Unsupported explicit town panel selection")
        elif self.kind in ("open_building", "construct", "select_recruit", "recruit", "transfer_stack", "select_hero", "hire_hero"):
            _text(self.selection)
            if self.panel is not None:
                raise ValueError("This action selects a displayed entry, not a panel")
        elif self.panel is not None or self.selection is not None:
            raise ValueError("This action has no panel selection")
        if self.kind in ("recruit", "transfer_stack"):
            _integer(self.count, low=1, high=2**31 - 1, name="explicit creature count")
            _text(self.destination)
        elif self.count is not None or self.destination is not None:
            raise ValueError("Only recruitment or transfer supplies a count and destination")


BUILDING_STATUSES = ("have_capital", "no_water", "forbidden", "add_mages_guild", "already_present",
                     "cant_build_today", "no_resources", "allowed", "prerequires", "missing_base")
_BUILDING_TEXT_ID = re.compile(r"building\.core\.[A-Za-z][A-Za-z0-9_]{0,63}\.[A-Za-z][A-Za-z0-9_]{0,63}\.(name|description)")


def _building_text(value, suffix):
    if type(value) is not str or _BUILDING_TEXT_ID.fullmatch(value) is None or not value.endswith("." + suffix):
        raise ValueError("Expected a static core building text ID")


@dataclass(frozen=True)
class Requirement:
    """Only the minimized requirement expression displayed in an open detail."""
    op: str
    name_text_id: str | None = None
    children: tuple[Requirement, ...] = ()

    def __post_init__(self):
        _tuple(self.children, Requirement, "requirement children")
        if self.op == "building":
            _building_text(self.name_text_id, "name")
            if self.children:
                raise ValueError("A requirement leaf has no children")
        elif self.op in ("all", "any", "none"):
            if self.name_text_id is not None:
                raise ValueError("A requirement expression has no building name")
        else:
            raise ValueError("Unsupported displayed requirement operation")
        pending = [(self, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 128 or depth > 16:
                raise ValueError("Displayed requirement expression exceeds bounds")
            pending.extend((child, depth + 1) for child in node.children)


@dataclass(frozen=True)
class PanelControl:
    panel: str
    ref: str

    def __post_init__(self):
        if self.panel not in ("hall", "recruitment", "tavern"):
            raise ValueError("Unsupported town navigation control")
        _text(self.ref)


@dataclass(frozen=True)
class BuildingChoice:
    ref: str
    name_text_id: str
    status: str

    def __post_init__(self):
        _text(self.ref)
        _building_text(self.name_text_id, "name")
        if self.status not in BUILDING_STATUSES:
            raise ValueError("Unsupported displayed construction status")


@dataclass(frozen=True)
class BuildingDetails:
    ref: str
    name_text_id: str
    description_text_id: str
    status: str
    cost: tuple[int, ...]
    requirements: Requirement | None = None

    def __post_init__(self):
        BuildingChoice(self.ref, self.name_text_id, self.status)
        _building_text(self.description_text_id, "description")
        if self.description_text_id[:-11] != self.name_text_id[:-4]:
            raise ValueError("Building name and description must identify the same definition")
        _tuple(self.cost, int, "construction cost")
        if len(self.cost) != 7:
            raise ValueError("Construction cost requires seven resource amounts")
        for amount in self.cost:
            _integer(amount, high=2**31 - 1)
        if self.status in ("prerequires", "missing_base"):
            if type(self.requirements) is not Requirement:
                raise ValueError("Displayed prerequisite status requires its expression")
            if self.status == "missing_base" and self.requirements.op != "building":
                raise ValueError("Missing base displays one building name")
        elif self.requirements is not None:
            raise ValueError("This detail status does not display prerequisites")


@dataclass(frozen=True)
class RecruitChoice:
    ref: str
    creature: int

    def __post_init__(self):
        _text(self.ref)
        # This is a static creature definition ID, as in a visible ArmyStack.
        _integer(self.creature, high=2**31 - 1)


@dataclass(frozen=True)
class RecruitDestination:
    ref: str
    role: str

    def __post_init__(self):
        _text(self.ref)
        if self.role not in ("town_garrison", "garrison_hero"):
            raise ValueError("Recruitment targets only the visible town upper army")


@dataclass(frozen=True)
class RecruitmentPanel:
    control_ref: str
    offers: tuple[RecruitChoice, ...]
    selected_offer_ref: str
    stock: int
    unit_cost: tuple[int, ...]
    slider_max: int
    destination: RecruitDestination

    def __post_init__(self):
        _text(self.control_ref)
        _tuple(self.offers, RecruitChoice, "recruitment cards")
        if not 1 <= len(self.offers) <= 32:
            raise ValueError("Recruitment requires bounded displayed creature cards")
        _unique([offer.ref for offer in self.offers], "creature offer")
        _unique([offer.creature for offer in self.offers], "displayed creature")
        if self.selected_offer_ref not in {offer.ref for offer in self.offers}:
            raise ValueError("Selected creature must be a displayed card")
        _integer(self.stock, high=2**31 - 1, name="available creatures")
        _integer(self.slider_max, high=self.stock, name="recruitment slider maximum")
        _tuple(self.unit_cost, int, "selected creature unit cost")
        if len(self.unit_cost) != 7:
            raise ValueError("Unit cost requires seven resource amounts")
        for amount in self.unit_cost:
            _integer(amount, high=2**31 - 1)
        if type(self.destination) is not RecruitDestination:
            raise ValueError("Recruitment requires the typed upper-army destination")


_HERO_NAME_TEXT_ID = re.compile(r"hero\.core\.[A-Za-z][A-Za-z0-9_]{0,63}\.name")
HERO_RECRUIT_STATUSES = ("allowed", "no_gold", "total_limit", "map_limit", "visitor_present", "no_offer")


@dataclass(frozen=True)
class HeroRecruitChoice:
    """A displayed tavern portrait/name card, never a hero object query."""
    ref: str
    portrait: int
    name_text_id: str

    def __post_init__(self):
        _text(self.ref)
        _integer(self.portrait, high=2**31 - 1)
        if type(self.name_text_id) is not str or _HERO_NAME_TEXT_ID.fullmatch(self.name_text_id) is None:
            raise ValueError("Expected a static core hero name text ID")


@dataclass(frozen=True)
class HeroRecruitDetails:
    """Only the description displayed for the selected tavern portrait."""
    level: int
    hero_class: int
    artifact_count: int

    def __post_init__(self):
        _integer(self.level, low=1, high=2**31 - 1)
        _integer(self.hero_class, high=2**31 - 1)
        _integer(self.artifact_count, high=2**31 - 1)


@dataclass(frozen=True)
class TavernPanel:
    offers: tuple[HeroRecruitChoice, ...]
    selected_offer_ref: str | None
    selected_detail: HeroRecruitDetails | None
    gold_cost: int
    recruit_status: str

    def __post_init__(self):
        _tuple(self.offers, HeroRecruitChoice, "tavern portrait cards")
        _unique([offer.ref for offer in self.offers], "tavern offer")
        _integer(self.gold_cost, high=2**31 - 1, name="displayed hero gold cost")
        if self.recruit_status not in HERO_RECRUIT_STATUSES:
            raise ValueError("Unsupported displayed hero recruitment status")
        if self.offers:
            if self.selected_offer_ref not in {offer.ref for offer in self.offers}:
                raise ValueError("Selected hero must be a displayed tavern card")
            if type(self.selected_detail) is not HeroRecruitDetails or self.recruit_status == "no_offer":
                raise ValueError("An offered hero requires its selected displayed description")
        elif self.selected_offer_ref is not None or self.selected_detail is not None or self.recruit_status == "allowed":
            raise ValueError("Empty tavern cannot contain a selected hero or enabled recruitment")


@dataclass(frozen=True)
class ArmySlot:
    ref: str
    slot: int
    creature: int | None
    count: int

    def __post_init__(self):
        _text(self.ref)
        _integer(self.slot, high=6)
        _integer(self.count, high=2**31 - 1, name="displayed stack count")
        if self.creature is None:
            if self.count != 0:
                raise ValueError("Empty army slot must have zero count")
        else:
            _integer(self.creature, high=2**31 - 1)
            if self.count == 0:
                raise ValueError("Occupied army slot requires a positive count")


@dataclass(frozen=True)
class ArmyView:
    ref: str
    role: str
    subject: str
    requires_nonempty: bool
    slots: tuple[ArmySlot, ...]

    def __post_init__(self):
        _text(self.ref)
        _text(self.subject)
        if self.role not in ("town_garrison", "garrison_hero", "visiting_hero"):
            raise ValueError("Unsupported displayed town army role")
        if type(self.requires_nonempty) is not bool or self.requires_nonempty != (self.role != "town_garrison"):
            raise ValueError("Last-stack rule must match the displayed army role")
        _tuple(self.slots, ArmySlot, "displayed army slots")
        if tuple(slot.slot for slot in self.slots) != tuple(range(7)):
            raise ValueError("Display exactly seven ordered army slots")
        _unique([self.ref, *(slot.ref for slot in self.slots)], "army/slot reference")
        if self.requires_nonempty and not any(slot.count for slot in self.slots):
            raise ValueError("A displayed hero must retain a nonempty army")


@dataclass(frozen=True)
class ArmyExchange:
    source: ArmyView
    destination: ArmyView | None

    def __post_init__(self):
        if type(self.source) is not ArmyView or self.source.role not in ("town_garrison", "garrison_hero"):
            raise ValueError("Transfer source is the native upper town army")
        if self.destination is not None:
            if type(self.destination) is not ArmyView or self.destination.role != "visiting_hero":
                raise ValueError("Transfer destination is only the actual own visiting hero")
            if self.source.subject == self.destination.subject:
                raise ValueError("Transfer armies must be distinct owned objects")
            _unique([army.ref for army in (self.source, self.destination)]
                    + [slot.ref for army in (self.source, self.destination) for slot in army.slots], "displayed army reference")

    def maximum(self, source_ref, destination_ref):
        """Exact native representability/last-stack rules; no strategic ranking."""
        if self.destination is None:
            return 0
        source = next((slot for slot in self.source.slots if slot.ref == source_ref), None)
        destination = next((slot for slot in self.destination.slots if slot.ref == destination_ref), None)
        if source is None or destination is None or source.count == 0 or destination.creature not in (None, source.creature):
            return 0
        # Both the native split UI and server sum both stacks in signed int32,
        # even if the selected transfer is small. Never offer an unsafe pair.
        if source.count + destination.count > 2**31 - 1:
            return 0
        reserve = int(self.source.requires_nonempty and sum(bool(slot.count) for slot in self.source.slots) == 1)
        return source.count - reserve

    def choices(self):
        if self.destination is None:
            return ()
        return tuple((source.ref, destination.ref, self.maximum(source.ref, destination.ref))
                     for source in self.source.slots for destination in self.destination.slots
                     if self.maximum(source.ref, destination.ref) > 0)


@dataclass(frozen=True)
class TownPanel:
    kind: str
    controls: tuple[PanelControl, ...] = ()
    buildings: tuple[BuildingChoice, ...] = ()
    building: BuildingDetails | None = None
    recruitment: RecruitmentPanel | None = None
    armies: ArmyExchange | None = None
    tavern: TavernPanel | None = None

    def __post_init__(self):
        _tuple(self.controls, PanelControl, "town controls")
        _tuple(self.buildings, BuildingChoice, "hall buildings")
        _unique([control.ref for control in self.controls], "panel control")
        _unique([choice.ref for choice in self.buildings], "building choice")
        if len(self.controls) > 32 or len(self.buildings) > 128:
            raise ValueError("Town panel exceeds display bounds")
        if self.kind == "overview":
            if self.buildings or self.building is not None:
                raise ValueError("Overview cannot expose hall or building details")
        elif self.kind == "hall":
            if self.controls or self.building is not None:
                raise ValueError("Hall exposes only its displayed choices")
        elif self.kind == "building":
            if self.controls or self.buildings or type(self.building) is not BuildingDetails:
                raise ValueError("Building panel exposes only its selected detail")
        elif self.kind == "recruitment":
            if self.controls or self.buildings or self.building is not None or type(self.recruitment) is not RecruitmentPanel:
                raise ValueError("Recruitment exposes only its selected dwelling panel")
        elif self.kind == "tavern":
            if self.controls or self.buildings or self.building is not None or type(self.tavern) is not TavernPanel:
                raise ValueError("Tavern exposes only its displayed portrait cards and selected description")
        else:
            raise ValueError("Unsupported town panel payload")
        if self.kind != "recruitment" and self.recruitment is not None:
            raise ValueError("Recruitment details require an open recruitment panel")
        if self.kind != "tavern" and self.tavern is not None:
            raise ValueError("Tavern details require an open tavern panel")
        if self.armies is not None and (self.kind != "overview" or type(self.armies) is not ArmyExchange):
            raise ValueError("Both armies are visible only in the open town overview")


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
    army: tuple[Stack, ...] | None
    primary_skills: tuple[int, int, int, int] | None = None
    mana: int | None = None
    level: int | None = None
    buildings: tuple[int, ...] | None = None
    faction: int | None = None
    town_panel: TownPanel | None = None

    def __post_init__(self):
        if type(self.context) is not Context or self.context.kind == "map":
            raise ValueError("Window requires a hero or town context")
        overlay = self.context.kind == "town" and self.context.panel in ("hall", "building", "recruitment", "tavern")
        if overlay:
            if self.army is not None or self.faction is not None:
                raise ValueError("Town overlays cannot refresh overview army or faction")
        else:
            _tuple(self.army, Stack, "army")
            _unique([s.slot for s in self.army], "army slot")
        if self.context.kind == "hero":
            if self.buildings is not None or self.faction is not None or self.town_panel is not None:
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
            if not overlay:
                _integer(self.faction)
            if self.context.panel is not None:
                if type(self.town_panel) is not TownPanel or self.town_panel.kind != self.context.panel:
                    raise ValueError("Town context requires its matching explicit panel")
                if self.context.panel == "building" and self.context.selection != self.town_panel.building.ref:
                    raise ValueError("Selected building does not match detail payload")
                if self.context.panel == "recruitment" and self.context.selection != self.town_panel.recruitment.control_ref:
                    raise ValueError("Selected dwelling does not match recruitment payload")
            elif self.town_panel is not None:
                raise ValueError("Legacy town context cannot expose deeper panels")


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
            if self.window.context.panel == "recruitment":
                panel = self.window.town_panel.recruitment
                # Pinned CCreature::maxAmount starts at this native integer cap
                # and the UI limits it by shared stock. Capacity is not a term.
                maximum = min(panel.stock, 2147483645,
                              *(amount // price for amount, price in zip(self.resources, panel.unit_cost) if price))
                if panel.slider_max != maximum:
                    raise ValueError("Recruitment slider must equal displayed stock/resource affordability")
            panel = self.window.town_panel
            if panel is not None and panel.tavern is not None:
                tavern = panel.tavern
                if tavern.recruit_status == "allowed" and self.resources[6] < tavern.gold_cost:
                    raise ValueError("Allowed hero recruitment must be affordable from visible gold")
                if tavern.recruit_status == "no_gold" and self.resources[6] >= tavern.gold_cost:
                    raise ValueError("Insufficient-gold status must match visible resources")
            if panel is not None and panel.armies is not None:
                exchange = panel.armies
                source = exchange.source
                if source.role == "town_garrison":
                    if source.subject != self.window.context.subject:
                        raise ValueError("Upper town army must bind the open town")
                elif source.subject not in {hero.handle for hero in self.heroes}:
                    raise ValueError("Garrisoned hero must be in this player's owned list")
                if exchange.destination is not None and exchange.destination.subject not in {hero.handle for hero in self.heroes}:
                    raise ValueError("Visiting hero must be in this player's owned list")
                upper = tuple(Stack(slot.slot, slot.creature, slot.count) for slot in source.slots if slot.count)
                if upper != self.window.army:
                    raise ValueError("Displayed upper army snapshots disagree")

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
    panel: TownPanel | None = None
    schema: str = ACTION_SCHEMA

    def __post_init__(self):
        profile = profile_for(self.schema)
        if self.schema != profile.action_schema:
            raise ValueError("Offers require an action schema")
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
            raise ValueError("Window support cannot expose map subject choices")
        _validate_context(self.context, profile)
        if self.context.kind == "town" and profile.town_economy:
            if type(self.panel) is not TownPanel or self.panel.kind != self.context.panel:
                raise ValueError("Town support must bind the displayed panel")
            _validate_panel(self.panel, profile)
        elif self.panel is not None:
            raise ValueError("This interface/context has no town panel support")

    @classmethod
    def from_observation(cls, observation, *, schema=ACTION_SCHEMA):
        if type(observation) is not Observation:
            raise ValueError("Expected typed observation")
        if observation.context.kind != "map":
            return cls(observation.context, observation.size, panel=observation.window.town_panel, schema=schema)
        return cls(Context(), observation.size, tuple(h.handle for h in observation.heroes),
                   tuple(t.handle for t in observation.towns), schema=schema)

    def permits(self, action):
        if type(action) is not Action or action.schema != self.schema:
            return False
        if self.panel is not None:
            if self.context.panel == "overview":
                if action.kind == "transfer_stack":
                    return (action.subject == self.context.subject and self.panel.armies is not None
                            and action.count <= self.panel.armies.maximum(action.selection, action.destination))
                return action.kind == "close" or (action.kind == "open_panel"
                    and action.subject == self.context.subject
                    and any(control.panel == action.panel and (action.panel in ("hall", "tavern") or control.ref == action.selection)
                            for control in self.panel.controls))
            if action.subject != self.context.subject:
                return False
            if action.kind == "back":
                return True
            if self.context.panel == "hall":
                return action.kind == "open_building" and action.selection in {item.ref for item in self.panel.buildings}
            if self.context.panel == "tavern":
                tavern = self.panel.tavern
                if action.kind == "select_hero":
                    return action.selection in {item.ref for item in tavern.offers}
                return (action.kind == "hire_hero" and action.selection == tavern.selected_offer_ref
                        and tavern.recruit_status == "allowed")
            if self.context.panel == "recruitment":
                recruitment = self.panel.recruitment
                if action.kind == "select_recruit":
                    return action.selection in {item.ref for item in recruitment.offers}
                return (action.kind == "recruit" and action.selection == recruitment.selected_offer_ref
                        and action.count <= recruitment.slider_max and action.destination == recruitment.destination.ref)
            return (action.kind == "construct" and action.selection == self.panel.building.ref
                    and self.panel.building.status == "allowed")
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
                               "interaction", "hero_unavailable", "opened", "closed", "turn_ended", "rejected", "built",
                               "selected", "recruited", "no_capacity", "transferred", "hired"):
            raise ValueError("Unsupported action result")
        if type(self.accepted) is not bool:
            raise ValueError("Result requires explicit action acceptance")
        if self.status in ("rejected", "no_capacity") and self.accepted:
            raise ValueError("Rejected result cannot be accepted")
        if self.status in ("opened", "closed", "turn_ended", "reached", "movement_limit", "no_route_known", "interaction", "built", "selected", "recruited", "transferred", "hired") and not self.accepted:
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
    *(f"core.genrltxt.{index}" for index in (1, 5, 7, 8, 30, 659)),
    *(f"core.arraytxt.{index}" for index in (*range(32, 42), *range(43, 58), *range(130, 136))),
})
TOWN_NOTICE_TEXT_IDS = frozenset({"core.genrltxt.17"})
# VCMI CCreature::getName{Singular,Plural}TextID uses the plural "creatures"
# namespace. The producer must still prove the delivered template/operation
# sequence and that the referenced creature is legitimate player information.
_CREATURE_NOTICE_TEXT_ID = re.compile(r"creatures\.core\.[A-Za-z][A-Za-z0-9_]{0,63}\.name\.(?:singular|plural)")
_NOTICE_CONTENT_ID = re.compile(r"core:[A-Za-z][A-Za-z0-9_]{0,63}")
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
                if (type(self.text_id) is not str or self.text_id not in NOTICE_TEXT_IDS | TOWN_NOTICE_TEXT_IDS
                        and _CREATURE_NOTICE_TEXT_ID.fullmatch(self.text_id) is None
                        and _HERO_NAME_TEXT_ID.fullmatch(self.text_id) is None):
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
    # Static artifact/spell definition shown by a delivered trophy component;
    # never an instance identifier, inventory query or description/utility text.
    content_id: str | None = None

    def __post_init__(self):
        if self.kind in ("resource", "resource_per_day"):
            if self.resource not in NOTICE_RESOURCES or self.owner is not None or self.content_id is not None:
                raise ValueError("Resource notice component requires only a known resource and amount")
            _integer(self.amount, low=-(2**31), high=2**31 - 1, name="notice resource amount")
        elif self.kind == "flag":
            if (self.resource is not None or self.amount is not None or self.content_id is not None
                    or self.owner not in ("self", "opponent")):
                raise ValueError("Flag notice component requires only a relative player")
        elif self.kind in ("artifact", "spell_scroll"):
            if (any(value is not None for value in (self.resource, self.amount, self.owner))
                    or type(self.content_id) is not str or _NOTICE_CONTENT_ID.fullmatch(self.content_id) is None):
                raise ValueError("Trophy notice component requires only a static core content ID")
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


def _no_capacity_notice(notice):
    return (type(notice) is Notice and notice.display_mode == "modal" and not notice.components
            and notice.text == (NoticeToken("append_text", text_id="core.genrltxt.17"),))


def _validate_notice(notice, profile):
    additional_hero_names = tuple(token for token in notice.text
                                 if token.text_id is not None and _HERO_NAME_TEXT_ID.fullmatch(token.text_id)
                                 and token.text_id not in NOTICE_TEXT_IDS)
    if additional_hero_names and (not profile.hero_recruitment or notice.display_mode != "modal"
            or notice.components or len(notice.text) != 2
            or notice.text[0] != NoticeToken("append_text", text_id="core.genrltxt.1")
            or notice.text[1].op != "replace_text" or len(additional_hero_names) != 1
            or notice.text[1] != additional_hero_names[0]):
        raise ValueError("Recruited-hero name requires the exact v5 level-cap notice")
    if any(token.text_id in TOWN_NOTICE_TEXT_IDS for token in notice.text):
        if not profile.town_economy or not _no_capacity_notice(notice):
            raise ValueError("Town capacity notice requires its exact v3-only displayed template")


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


def _validate_context(context, profile):
    if profile == WINDOWS_PROFILE and (context.panel is not None or context.selection is not None):
        raise ValueError("History v2 cannot contain town panels")
    if context.panel == "tavern" and not profile.hero_recruitment:
        raise ValueError("Tavern context requires history v5")
    if profile.town_economy and context.kind == "town" and context.panel is None:
        raise ValueError("Town economy history requires an explicit town panel")


def _validate_panel(panel, profile):
    if not profile.hero_recruitment and (panel.tavern is not None or any(control.panel == "tavern" for control in panel.controls)):
        raise ValueError("Tavern payload and controls require history v5")
    expected = profile.army_transfer and panel.kind == "overview"
    if expected != (panel.armies is not None):
        raise ValueError("Explicit displayed armies belong only to v4 town overview")


def _validate_observation(observation, profile):
    _validate_context(observation.context, profile)
    if profile == WINDOWS_PROFILE and observation.window is not None and observation.window.town_panel is not None:
        raise ValueError("History v2 cannot contain town economy data")
    if observation.window is not None and observation.window.town_panel is not None:
        _validate_panel(observation.window.town_panel, profile)


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
        profile = profile_for(self.schema)
        if self.schema != profile.history_schema or self.kind not in _PAYLOADS:
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
        if self.context is not None:
            _validate_context(self.context, profile)
        if self.observation is not None:
            _validate_observation(self.observation, profile)
        if self.action is not None and self.action.schema != profile.action_schema:
            raise ValueError("Action/history schema mismatch")
        if self.result is not None and profile == WINDOWS_PROFILE and self.result.status in ("built", "selected", "recruited", "no_capacity"):
            raise ValueError("History v2 cannot contain town economy results")
        if self.result is not None and not profile.army_transfer and self.result.status == "transferred":
            raise ValueError("Army transfer results require history v4")
        if self.result is not None and not profile.hero_recruitment and self.result.status == "hired":
            raise ValueError("Hero recruitment results require history v5")
        if self.event is not None and self.event.notice is not None:
            _validate_notice(self.event.notice, profile)
        if self.kind == "decision" and self.offers != Offers.from_observation(self.observation, schema=profile.action_schema):
            raise ValueError("Offers must preserve complete context and all coordinate support")


@dataclass(frozen=True)
class TerminalClosure:
    """References to a verified episode ending, never a native action result.

    Consumption establishes whose last action was being handled. A terminal
    closure does not assert issuance, acknowledgement or command success.
    Full paired-prefix/manifest verification remains the collector's duty.
    """
    consumed_seq: int
    superseded_seq: int
    terminal_seq: int
    finalized_seq: int

    def __post_init__(self):
        previous = -1
        for value in (self.consumed_seq, self.superseded_seq, self.terminal_seq, self.finalized_seq):
            _integer(value, low=previous + 1)
            previous = value


@dataclass(frozen=True)
class Transition:
    episode: str
    player: int
    decision_seq: int
    action: Action
    result_seq: int | None
    successor_seq: int | None
    reward: int | None
    eligible: bool
    bootstrap_allowed: bool
    boundary: str
    terminal_closure: TerminalClosure | None = None


class PlayerHistory:
    """Append-only one-player stream. It does not allocate or update neural state."""

    def __init__(self, episode: str, player: int, *, schema=SCHEMA):
        _text(episode, "episode")
        _integer(player, high=1)
        self.episode, self.player = episode, player
        self.profile = profile_for(schema)
        if schema != self.profile.history_schema:
            raise ValueError("PlayerHistory requires a history schema")
        self.schema = schema
        self._frames = []
        self.context = Context()
        self.revision = -1
        self.day = 1
        self.decision = None
        self.action = None
        self.consumed = False
        self.consumer = None
        self.capacity_notice = False
        self.context_changes = 0
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
        frame = Frame(self.episode, self.player, len(self._frames), self.day if day is None else day, kind,
                      schema=self.schema, **payload)
        self.append(frame)
        return frame

    def append(self, frame):
        if type(frame) is not Frame or (frame.episode, frame.player, frame.schema) != (self.episode, self.player, self.schema):
            raise ValueError("Cross-player/episode/schema or untyped frame")
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
            if self.action.kind in ("construct", "transfer_stack", "hire_hero") and frame.consumer != "engine":
                raise ValueError("Construction, transfer and hero hiring require native engine consumption")
            self.consumed = True
            self.consumer = frame.consumer
        elif kind == "issued":
            self._match_consumed(frame)
            if self.action.kind not in ("move_to", "end_turn", "construct", "recruit", "transfer_stack", "hire_hero"):
                raise ValueError("Navigation cannot issue native game commands")
            if self.action.kind == "recruit" and (self.consumer != "engine" or self.capacity_notice):
                raise ValueError("Recruitment issuance requires native consumption without capacity rejection")
            if (self.pending_command is not None or frame.command_index != self.commands or self.native_rejected
                    or self.action.kind in ("end_turn", "construct", "recruit", "transfer_stack", "hire_hero") and self.commands != 0):
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
            expected = self._context_target()
            if expected != frame.context or frame.revision <= self.revision:
                raise ValueError("Context change must match consumed navigation and advance revision")
            if self.action.kind in ("construct", "recruit", "hire_hero") and self.commands != 1:
                raise ValueError("Purchase closes its panel only after genuine native issuance")
            if self.action.kind in ("open_panel", "open_building", "back", "construct", "select_recruit", "recruit", "select_hero", "hire_hero") and self.context_changes:
                raise ValueError("Town action has already emitted its context transition")
            self.context, self.revision = frame.context, frame.revision
            self.context_changes += 1
        elif kind == "event" and frame.event.notice is not None and _no_capacity_notice(frame.event.notice):
            if (self.action is None or self.action.kind != "recruit" or not self.consumed
                    or self.consumer != "adapter" or self.commands != 0 or self.capacity_notice
                    or self.context != self.decision.observation.context):
                raise ValueError("Capacity notice requires one consumed, unissued recruitment attempt")
            self.capacity_notice = True
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
        self.consumer, self.capacity_notice = None, False
        self.context_changes = 0
        self.native_rejected = False

    def _context_target(self):
        action = self.action
        before = self.decision.observation.context
        if action.kind == "open_hero":
            return Context("hero", action.subject)
        if action.kind == "open_town":
            return Context("town", action.subject, "overview" if self.profile.town_economy else None)
        if action.kind == "close":
            return Context()
        if action.kind == "open_panel":
            return Context("town", action.subject, action.panel, action.selection)
        if action.kind == "open_building":
            return Context("town", action.subject, "building", action.selection)
        if action.kind == "back":
            return Context("town", action.subject, "hall" if before.panel == "building" else "overview")
        if action.kind in ("construct", "recruit", "hire_hero"):
            return Context("town", action.subject, "overview")
        if action.kind in ("select_recruit", "select_hero"):
            return before
        return None

    def _check_result(self, result):
        action = self.action
        if action.kind == "transfer_stack":
            self._check_transfer_result(result)
        elif action.kind == "move_to":
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
            expected = {"open_hero": "opened", "open_town": "opened", "close": "closed", "end_turn": "turn_ended",
                        "open_panel": "opened", "open_building": "opened", "back": "closed", "construct": "built",
                        "select_recruit": "selected", "recruit": "recruited", "select_hero": "selected", "hire_hero": "hired"}
            if result.status not in (expected[action.kind], "rejected", *( ("no_capacity",) if action.kind == "recruit" else () )):
                raise ValueError("Incorrect contextual result")
            target = self._context_target()
            if action.kind == "end_turn" and result.accepted and (self.commands != 1 or self.accepted_commands != 1):
                raise ValueError("Successful turn ending requires exactly one accepted native command")
            if action.kind == "end_turn" and not result.accepted and self.accepted_commands != 0:
                raise ValueError("Accepted native turn ending cannot be reported as rejected")
            if action.kind == "recruit" and result.status == "no_capacity":
                observed = next((frame.observation for frame in reversed(self._frames)
                                 if frame.observation is not None), self.decision.observation)
                if (self.consumer != "adapter" or self.commands != 0 or self.accepted_commands != 0
                        or not self.capacity_notice or self.context_changes != 0
                        or self.context != self.decision.observation.context
                        or observed.resources != self.decision.observation.resources
                        or observed.window != self.decision.observation.window):
                    raise ValueError("No-capacity result requires its visible UI rejection without native issuance")
            elif action.kind in ("construct", "recruit", "hire_hero"):
                if (self.commands != 1 or self.accepted_commands != int(result.accepted)
                        or self.context != target or self.capacity_notice):
                    raise ValueError("Purchase requires one native acknowledgement and explicit overview return")
            if action.kind == "hire_hero":
                self._check_hire_effect(result)
            if action.kind in ("select_recruit", "select_hero"):
                observed = next((frame.observation for frame in reversed(self._frames)
                                 if frame.observation is not None), self.decision.observation)
                if self.consumer != "adapter" or observed.context != self.context:
                    raise ValueError("Card selection requires adapter consumption in the same panel")
                panel = observed.window.town_panel
                selected = panel.tavern if action.kind == "select_hero" else panel.recruitment
                if result.accepted and selected.selected_offer_ref != action.selection:
                    raise ValueError("Selection result must match the visible selected card")
                if action.kind == "select_hero" and result.accepted:
                    before = self.decision.observation
                    old = before.window.town_panel.tavern
                    if (observed.resources != before.resources or observed.heroes != before.heroes
                            or observed.towns != before.towns or selected.offers != old.offers
                            or selected.gold_cost != old.gold_cost or selected.recruit_status != old.recruit_status):
                        raise ValueError("Selecting a tavern card cannot change game state or unselected cards")
                if not result.accepted and (self.context_changes != 0 or self.revision != self.decision.revision
                                            or observed.window != self.decision.observation.window):
                    raise ValueError("Rejected selection cannot change the displayed card or context revision")
            if result.accepted and action.kind != "end_turn" and self.context != target:
                raise ValueError("Navigation result requires matching explicit context change")
            if result.accepted and action.kind in ("open_panel", "open_building", "back", "construct", "select_recruit", "recruit", "select_hero", "hire_hero") and self.context_changes != 1:
                raise ValueError("Successful town operation requires its explicit context revision")
            if not result.accepted and action.kind not in ("construct", "recruit", "hire_hero") and self.context != self.decision.observation.context:
                raise ValueError("Rejected navigation cannot have changed context")

    def _check_hire_effect(self, result):
        before = self.decision.observation
        observed = next((frame.observation for frame in reversed(self._frames)
                         if frame.observation is not None), before)
        tavern = before.window.town_panel.tavern
        expected_resources = list(before.resources)
        if result.accepted:
            expected_resources[6] -= tavern.gold_cost
        prior = {hero.handle: hero for hero in before.heroes}
        current = {hero.handle: hero for hero in observed.heroes}
        if (observed.resources != tuple(expected_resources) or observed.towns != before.towns
                or any(current.get(ref) != hero for ref, hero in prior.items())
                or len(current) != len(prior) + int(result.accepted)):
            raise ValueError("Hero recruitment must match exact gold cost and one newly owned hero")
        # The selected native hero definition and visiting placement are proved
        # by native receipts/replay; opaque handles do not encode that identity.

    def _check_transfer_result(self, result):
        if result.status not in ("transferred", "rejected") or any(value is not None for value in (
                result.position, result.movement, result.destination_reached)):
            raise ValueError("Transfer result cannot claim movement or another operation")
        before = self.decision.observation
        observed = next((frame.observation for frame in reversed(self._frames)
                         if frame.observation is not None), before)
        if (self.consumer != "engine" or self.commands != 1 or self.accepted_commands != int(result.accepted)
                or self.context_changes != 0 or self.revision != self.decision.revision
                or observed.context != before.context or self.context != before.context
                or observed.resources != before.resources):
            raise ValueError("Transfer requires one acknowledgement, unchanged overview and unchanged resources")
        expected = before.window
        if result.accepted:
            action = self.action
            exchange = expected.town_panel.armies
            if exchange.maximum(action.selection, action.destination) < action.count:
                raise ValueError("Transfer quantity exceeds the displayed exact native support")
            source = next(slot for slot in exchange.source.slots if slot.ref == action.selection)
            def changed(slot, *, receiving):
                count = slot.count + action.count if receiving else slot.count - action.count
                return replace(slot, count=count, creature=source.creature if count else None)
            upper = replace(exchange.source, slots=tuple(changed(slot, receiving=False) if slot.ref == action.selection
                                                        else slot for slot in exchange.source.slots))
            lower = replace(exchange.destination, slots=tuple(changed(slot, receiving=True) if slot.ref == action.destination
                                                             else slot for slot in exchange.destination.slots))
            panel = replace(expected.town_panel, armies=ArmyExchange(upper, lower))
            expected = replace(expected, town_panel=panel,
                               army=tuple(Stack(slot.slot, slot.creature, slot.count) for slot in upper.slots if slot.count))
        if observed.window != expected:
            raise ValueError("Transfer acknowledgement must match the exact displayed two-army effect")

    def _terminal_closure(self):
        """Derive closure only for the final consumed action, without new frames."""
        if self.boundary != "natural" or not self._frames or self._frames[-1].kind != "finalized":
            return None
        decisions = [f for f in self._frames if f.kind == "decision"]
        consumed = [f for f in self._frames if f.kind == "consumed"]
        if not decisions or not consumed or consumed[-1].decision_seq != decisions[-1].seq:
            return None
        decision, consumption = decisions[-1], consumed[-1]
        related = [f for f in self._frames if f.decision_seq == decision.seq]
        submissions = [f for f in related if f.kind == "submitted"]
        supersessions = [f for f in related if f.kind == "superseded"]
        terminals = [f for f in self._frames if f.kind == "terminal"]
        if (len(submissions) != 1 or len(supersessions) != 1 or len(terminals) != 1
                or supersessions[0].reason != "terminal" or any(f.kind == "result" for f in related)):
            return None
        closure = TerminalClosure(consumption.seq, supersessions[0].seq, terminals[0].seq, self._frames[-1].seq)
        return decision.seq, submissions[0].action, closure

    def transitions(self, *, schema=TRAJECTORY_SCHEMA):
        """No targets until boundary finalization; v1 excludes incomplete actions.

        Successors are subsequent decisions in THIS history, never opponent
        turns. This is environment provenance, not behavior probabilities, value
        estimates or a PPO rollout. V2 can close the last consumed action at a
        verified natural terminal without inventing an ordinary result. A failed
        record has no usable targets. History encodings are unchanged by either
        trajectory interpretation.
        """
        if schema not in (TRAJECTORY_SCHEMA, TERMINAL_TRAJECTORY_SCHEMA):
            raise ValueError("Unsupported contextual trajectory schema")
        completed = []
        actions = {}
        consumed = {f.decision_seq for f in self._frames if f.kind == "consumed"}
        withdrawn = {f.decision_seq for f in self._frames if f.kind == "superseded" and f.decision_seq not in consumed}
        decisions = [f.seq for f in self._frames if f.kind == "decision" and f.seq not in withdrawn]
        for frame in self._frames:
            if frame.kind == "submitted":
                actions[frame.decision_seq] = frame.action
            elif frame.kind == "result":
                completed.append((frame.decision_seq, actions[frame.decision_seq], frame.seq, None))
        closure = self._terminal_closure() if schema == TERMINAL_TRAJECTORY_SCHEMA else None
        if closure is not None:
            decision, action, evidence = closure
            completed.append((decision, action, None, evidence))
        eligible = self.boundary in ("natural", "cutoff")
        transitions = []
        for index, (decision, action, result, closure) in enumerate(completed):
            successor = next((seq for seq in decisions if seq > result), None) if result is not None else None
            terminal_owner = (self.boundary == "natural" and index == len(completed) - 1
                              and not any(seq > decision for seq in consumed))
            if terminal_owner:
                successor = None
            valid = eligible and (successor is not None or terminal_owner)
            reward = (1 if self.terminal == "win" else -1) if valid and terminal_owner else 0 if valid else None
            transitions.append(Transition(self.episode, self.player, decision, action, result, successor,
                                          reward, valid, valid and successor is not None,
                                          self.boundary or "pending", closure))
        return tuple(transitions)


def finalize_natural(histories, *, engine_outcomes, socket_outcomes, process_exit_code, evidence_ref,
                     histories_complete=(True, True)):
    """Finalize both seats after external evidence checks; never infer from exit alone."""
    if (type(histories) is not tuple or len(histories) != 2 or any(type(h) is not PlayerHistory for h in histories)
            or tuple(h.player for h in histories) != (0, 1) or histories[0].episode != histories[1].episode
            or histories[0].schema != histories[1].schema):
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
            or histories[0].schema != histories[1].schema
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


def _decode_context(raw, profile):
    data = _record(raw, Context)
    if profile == WINDOWS_PROFILE or data.get("kind", "map") != "town":
        if {"panel", "selection"} & data.keys():
            raise ValueError("Panel fields are absent from this context/schema")
    elif not {"panel", "selection"} <= data.keys():
        raise ValueError("Town v3 context requires explicit panel and selection")
    context = _decode_simple(data, Context)
    _validate_context(context, profile)
    return context


def _exact_record(raw, cls):
    data = _record(raw, cls)
    if set(data) != {field.name for field in fields(cls)}:
        raise ValueError(f"Incomplete {cls.__name__} wire payload")
    return data


def decode_town_panel(raw, *, schema=TOWN_SCHEMA):
    profile = profile_for(schema)
    if schema != profile.history_schema or not profile.town_economy:
        raise ValueError("Town panel requires its explicit history schema")
    data = _record(raw, TownPanel)
    expected = {"kind", "controls", "buildings", "building"}
    if data.get("kind") == "recruitment":
        expected.add("recruitment")
    if profile.army_transfer and data.get("kind") == "overview":
        expected.add("armies")
    if profile.hero_recruitment and data.get("kind") == "tavern":
        expected.add("tavern")
    if set(data) != expected:
        raise ValueError("Town panel payload must match its exact displayed panel")
    data["controls"] = tuple(_decode_simple(item, PanelControl) for item in _array(data["controls"]))
    data["buildings"] = tuple(_decode_simple(item, BuildingChoice) for item in _array(data["buildings"]))
    if data["building"] is not None:
        building = _exact_record(data["building"], BuildingDetails)
        building["cost"] = _array(building["cost"])
        nodes = 0

        def requirement(raw_node, depth=0):
            nonlocal nodes
            nodes += 1
            if nodes > 128 or depth > 16:
                raise ValueError("Displayed requirement expression exceeds bounds")
            node = _exact_record(raw_node, Requirement)
            node["children"] = tuple(requirement(child, depth + 1) for child in _array(node["children"]))
            return Requirement(**node)

        if building["requirements"] is not None:
            building["requirements"] = requirement(building["requirements"])
        data["building"] = BuildingDetails(**building)
    if "recruitment" in data:
        recruitment = _exact_record(data["recruitment"], RecruitmentPanel)
        recruitment["offers"] = tuple(_decode_simple(offer, RecruitChoice) for offer in _array(recruitment["offers"]))
        recruitment["unit_cost"] = _array(recruitment["unit_cost"])
        recruitment["destination"] = _decode_simple(recruitment["destination"], RecruitDestination)
        data["recruitment"] = RecruitmentPanel(**recruitment)
    if "tavern" in data:
        tavern = _exact_record(data["tavern"], TavernPanel)
        tavern["offers"] = tuple(HeroRecruitChoice(**_exact_record(offer, HeroRecruitChoice)) for offer in _array(tavern["offers"]))
        if tavern["selected_detail"] is not None:
            tavern["selected_detail"] = HeroRecruitDetails(**_exact_record(tavern["selected_detail"], HeroRecruitDetails))
        data["tavern"] = TavernPanel(**tavern)
    if "armies" in data:
        exchange = _exact_record(data["armies"], ArmyExchange)
        for key in ("source", "destination"):
            if exchange[key] is not None:
                army = _exact_record(exchange[key], ArmyView)
                army["slots"] = tuple(ArmySlot(**_exact_record(slot, ArmySlot)) for slot in _array(army["slots"]))
                exchange[key] = ArmyView(**army)
        data["armies"] = ArmyExchange(**exchange)
    return TownPanel(**data)


def decode_observation(raw, *, schema=SCHEMA):
    profile = profile_for(schema)
    if schema != profile.history_schema:
        raise ValueError("Observation decoding requires its enclosing history schema")
    data = _record(raw, Observation)
    for name in ("size", "resources"):
        if name in data:
            data[name] = _array(data[name])
    for name, cls in (("tiles", Tile), ("objects", ObjectView), ("heroes", Hero), ("towns", Town)):
        if name in data:
            data[name] = tuple(_decode_simple(item, cls, ("position",)) for item in _array(data[name]))
    if data.get("window") is not None:
        window = _record(data["window"], Window)
        window["context"] = _decode_context(window.get("context"), profile)
        if profile == WINDOWS_PROFILE or window["context"].kind != "town":
            if "town_panel" in window:
                raise ValueError("Town economy fields are absent from this context/schema")
        else:
            window["town_panel"] = decode_town_panel(window.get("town_panel"), schema=profile.history_schema)
        if window.get("army") is not None:
            window["army"] = tuple(_decode_simple(item, Stack) for item in _array(window["army"]))
        for name in ("primary_skills", "buildings"):
            if window.get(name) is not None:
                window[name] = _array(window[name])
        try:
            data["window"] = Window(**window)
        except TypeError as exc:
            raise ValueError("Incomplete window") from exc
    try:
        observation = Observation(**data)
        _validate_observation(observation, profile)
        return observation
    except TypeError as exc:
        raise ValueError("Incomplete observation") from exc


def decode_notice(raw, *, schema=SCHEMA):
    """Allowlisted JSON notice; unused pools/raw text/engine IDs are rejected."""
    data = _record(raw, Notice)
    data["text"] = tuple(_decode_simple(token, NoticeToken) for token in _array(data.get("text")))
    data["components"] = tuple(_decode_simple(component, NoticeComponent)
                               for component in _array(data.get("components")))
    try:
        notice = Notice(**data)
        profile = profile_for(schema)
        if schema != profile.history_schema:
            raise ValueError("Notice decoding requires its enclosing history schema")
        _validate_notice(notice, profile)
        return notice
    except TypeError as exc:
        raise ValueError("Incomplete informational notice") from exc


def _wire(value):
    """Detached canonical JSON, omitting additive fields from their older schema."""
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        data = {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
        if type(value) is Context and value.panel is None:
            del data["panel"], data["selection"]
        elif type(value) is Action and value.schema == ACTION_SCHEMA:
            for name in ("panel", "selection", "count", "destination"):
                del data[name]
        elif type(value) is Window and value.town_panel is None:
            del data["town_panel"]
        elif type(value) is Offers:
            del data["schema"]  # bound by the enclosing frame's explicit schema
            if value.panel is None:
                del data["panel"]
        elif type(value) is TownPanel:
            for name in ("recruitment", "armies", "tavern"):
                if getattr(value, name) is None:
                    del data[name]
        elif type(value) is NoticeComponent and value.content_id is None:
            del data["content_id"]
        return data
    return value


def encode_action(action):
    if type(action) is not Action:
        raise ValueError("Expected typed contextual action")
    return _wire(action)


def encode_observation(observation, *, schema=SCHEMA):
    if type(observation) is not Observation:
        raise ValueError("Expected typed contextual observation")
    profile = profile_for(schema)
    if schema != profile.history_schema:
        raise ValueError("Observation encoding requires its enclosing history schema")
    _validate_observation(observation, profile)
    return _wire(observation)


def encode_offers(offers):
    if type(offers) is not Offers:
        raise ValueError("Expected typed contextual offers")
    return _wire(offers)


def encode_frame(frame):
    if type(frame) is not Frame:
        raise ValueError("Expected typed contextual frame")
    return _wire(frame)


def decode_frame(raw):
    data = _record(raw, Frame)
    if not {"schema", "episode", "player", "seq", "day", "kind"} <= data.keys():
        raise ValueError("History JSON requires explicit schema and base fields")
    profile = profile_for(data["schema"])
    if data["schema"] != profile.history_schema:
        raise ValueError("Expected a history schema")
    if data.get("observation") is not None:
        data["observation"] = decode_observation(data["observation"], schema=profile.history_schema)
    if data.get("context") is not None:
        data["context"] = _decode_context(data["context"], profile)
    if data.get("action") is not None and profile == WINDOWS_PROFILE:
        if type(data["action"]) is not dict or {"panel", "selection", "count", "destination"} & data["action"].keys():
            raise ValueError("Action v2 cannot carry town economy fields")
    for name, cls in (("action", Action), ("result", Result)):
        if data.get(name) is not None:
            data[name] = _decode_simple(data[name], cls, ("position",) if name == "result" else ())
    if data.get("event") is not None:
        event = _record(data["event"], VisibleEvent)
        if event.get("notice") is not None:
            event["notice"] = decode_notice(event["notice"], schema=profile.history_schema)
        data["event"] = _decode_simple(event, VisibleEvent)
    if data.get("offers") is not None:
        offer = _record(data["offers"], Offers)
        if "schema" in offer or profile == WINDOWS_PROFILE and "panel" in offer:
            raise ValueError("Unsupported offer schema/panel wire field")
        offer["schema"] = profile.action_schema
        offer["context"] = _decode_context(offer.get("context"), profile)
        if "panel" in offer:
            offer["panel"] = decode_town_panel(offer["panel"], schema=profile.history_schema)
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
