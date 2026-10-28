#include "StdInc.h"
#include "ExplorerRL.h"
#include "VisibleWorld.h"
#include "ArtifactMoveRules.h"

#include "../../lib/GameLibrary.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/entities/artifact/ArtifactUtils.h"
#include "../../lib/entities/artifact/CArtHandler.h"
#include "../../lib/entities/artifact/CArtifact.h"
#include "../../lib/entities/artifact/CArtifactInstance.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGCreature.h"
#include "../../lib/gameState/InfoAboutArmy.h"
#include "../../lib/bonuses/BonusParameters.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/networkPacks/ArtifactLocation.h"
#include "../../lib/networkPacks/Component.h"
#include "../../lib/mapping/TerrainTile.h"
#include "../../lib/spells/CSpellHandler.h"
#include "../../lib/spells/Problem.h"
#include "../../lib/spells/ViewSpellInt.h"
#include "../../lib/spells/adventure/AdventureSpellEffect.h"

#include <limits>

namespace
{
using State = explorer_rl::ContextualState;
JsonNode choice(const std::string & kind, const std::string & subject)
{
	JsonNode result;
	for(const char * key : {"ref", "destination", "artifact", "slot", "destination_slot", "spell", "skill", "skill_level", "resource_from", "resource_to", "give", "receive", "max_count", "x", "y", "cost"}) result[key] = JsonNode();
	result["kind"].String() = kind;
	result["subject"].String() = subject;
	return result;
}
JsonNode item(const std::string & kind, const std::string & subject, int id)
{
	JsonNode result;
	for(const char * key : {"ref", "slot", "amount", "cost", "x", "y", "z", "spell", "owner"}) result[key] = JsonNode();
	result["kind"].String() = kind;
	result["subject"].String() = subject;
	result["id"].Integer() = id;
	result["locked"].Bool() = false;
	return result;
}
JsonNode panel(const std::string & ref, const std::string & kind, const JsonNode & parent, bool mandatory = false)
{
	JsonNode result;
	result["ref"].String() = ref;
	result["kind"].String() = kind;
	result["parent_context"] = parent;
	result["mandatory"].Bool() = mandatory;
	result["hero_level"] = JsonNode();
	result["primary_skill"] = JsonNode();
	result["inspected_subject"] = JsonNode();
	result["items"].Vector(); result["choices"].Vector();
	return result;
}
void append(JsonNode & panel, const char * field, JsonNode value)
{
	value["ref"].String() = panel["ref"].String() + "-" + field + "-" + std::to_string(panel[field].Vector().size());
	panel[field].Vector().push_back(std::move(value));
}
int resolve(const State & state, const std::string & subject)
{
	const auto found = state.subjects.find(subject);
	if(found == state.subjects.end()) throw std::runtime_error("Unknown displayed capability subject");
	return found->second;
}
const CGHeroInstance * hero(const CCallback & callback, const State & state, const std::string & subject)
{
	const auto * result = callback.getHero(ObjectInstanceID(resolve(state, subject)));
	if(!result || result->getOwner() != callback.getPlayerID()) throw std::runtime_error("Capability hero is no longer owned");
	return result;
}
const CGTownInstance * town(const CCallback & callback, const State & state, const std::string & subject)
{
	const auto * result = callback.getTown(ObjectInstanceID(resolve(state, subject)));
	if(!result || result->getOwner() != callback.getPlayerID()) throw std::runtime_error("Capability town is no longer owned");
	return result;
}
void change(State & state, const JsonNode & context, int day, const char * recordKind, int64_t decision = -1, const char * reason = nullptr)
{
	state.context = context["kind"].String();
	state.subject = context["subject"].isString() ? context["subject"].String() : "";
	state.panel = context["panel"].isString() ? context["panel"].String() : "";
	state.selection = context["selection"].isString() ? context["selection"].String() : "";
	auto record = state.record(recordKind, day);
	record["context"] = state.contextJson();
	record["revision"].Integer() = ++state.revision;
	if(decision >= 0) record["decision_id"].Integer() = decision;
	if(reason) record["reason"].String() = reason;
	state.records.push_back(std::move(record));
}
JsonNode mapContext()
{
	JsonNode result; result["kind"].String() = "map"; result["subject"] = JsonNode(); return result;
}
JsonNode cost(int64_t gold)
{
	JsonNode result; for(int i = 0; i < 7; ++i) result.Vector().emplace_back(i == 6 ? gold : 0); return result;
}
std::vector<const CGObjectInstance *> visionsTargets(const CCallback & callback, const CGHeroInstance & caster)
{
	std::vector<const CGObjectInstance *> result;
	std::set<int> seen;
	const auto size = callback.getMapSize();
	for(int z = 0; z < size.z; ++z) for(int y = 0; y < size.y; ++y) for(int x = 0; x < size.x; ++x)
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile) continue;
		for(const auto id : tile->visitableObjects)
		{
			const auto * object = explorer_rl::playerVisibleObject(callback, id);
			if(!object || !seen.insert(id.getNum()).second) continue;
			const auto subtype = object->ID == Obj::MONSTER ? BonusCustomSubtype::visionsMonsters
				: object->ID == Obj::HERO ? BonusCustomSubtype::visionsHeroes : BonusCustomSubtype::visionsTowns;
			if((object->ID == Obj::MONSTER || object->ID == Obj::HERO || object->ID == Obj::TOWN)
				&& caster.hasVisions(object, subtype)) result.push_back(object);
		}
	}
	return result;
}

std::vector<ArtifactPosition> slots(const CGHeroInstance & hero)
{
	auto result = ArtifactUtils::allWornSlots();
	for(size_t i = 0; i < hero.artifactsInBackpack.size(); ++i)
		result.emplace_back(ArtifactPosition::BACKPACK_START + static_cast<int>(i));
	return result;
}
}

JsonNode ExplorerRL::capabilityControls()
{
	JsonNode result; result.Vector();
	if(!expandedInterface) return result;
	if(contextState.context == "hero")
	{
		const auto * current = hero(*callback, contextState, contextState.subject);
		result.Vector().emplace_back("artifacts");
		if(current->hasSpellbook()) result.Vector().emplace_back("spellbook");
		if(current->hasBonusOfType(BonusType::VISIONS)) result.Vector().emplace_back("visions");
	}
	else if(contextState.context == "town" && contextState.panel == "overview")
	{
		const auto * current = town(*callback, contextState, contextState.subject);
		if(current->hasBuilt(BuildingID::MARKETPLACE)) result.Vector().emplace_back("marketplace");
		bool purchase = current->hasBuilt(BuildingID::MAGES_GUILD_1);
		for(const auto & artifact : LIBRARY->arth->objects)
			if(artifact && artifact->getWarMachine().hasValue() && current->isWarMachineAvailable(artifact->getId())) purchase = true;
		if(purchase) result.Vector().emplace_back("purchases");
	}
	else if(contextState.context == "exchange" && contextState.ownHeroExchange
		&& !contextState.ownHeroExchange->joiningNeutral && !contextState.ownHeroExchange->objectGarrison)
		result.Vector().emplace_back("artifacts");
	return result;
}

JsonNode ExplorerRL::capabilityPanel()
{
	if(contextState.context != "capability") throw std::runtime_error("Capability contents require an opened panel");
	if(contextState.capabilityDialog && contextState.capabilityDialog->installed)
		return contextState.capabilityDialog->panel;
	auto result = panel(contextState.capabilityRef, contextState.panel, contextState.capabilityParentContext);
	const auto & subject = contextState.subject;
	if(contextState.panel == "marketplace")
	{
		const auto * current = town(*callback, contextState, subject);
		if(!current->hasBuilt(BuildingID::MARKETPLACE) || !current->allowsTrade(EMarketMode::RESOURCE_RESOURCE)) return result;
		for(int from = 0; from < 7; ++from) for(int to = 0; to < 7; ++to)
		{
			int give = 0, receive = 0;
			if(from == to || !current->getOffer(from, to, give, receive, EMarketMode::RESOURCE_RESOURCE) || give <= 0 || receive <= 0) continue;
			const int64_t maximum = std::min<int64_t>(callback->getResourceAmount(GameResID(from)) / give, std::numeric_limits<int32_t>::max() / give);
			if(maximum < 1) continue;
			auto selected = choice("trade_resource", subject);
			selected["resource_from"].Integer() = from; selected["resource_to"].Integer() = to;
			selected["give"].Integer() = give; selected["receive"].Integer() = receive; selected["max_count"].Integer() = maximum;
			append(result, "choices", std::move(selected));
		}
	}
	else if(contextState.panel == "purchases")
	{
		const auto * current = town(*callback, contextState, subject);
		for(const auto * owner : {current->getVisitingHero(), current->getGarrisonHero()})
		{
			if(!owner || owner->getOwner() != playerID) continue;
			for(const auto & artifact : LIBRARY->arth->objects)
			{
				if(!artifact) continue;
				const auto id = artifact->getId();
				const bool book = id == ArtifactID::SPELLBOOK;
				if(book ? !current->hasBuilt(BuildingID::MAGES_GUILD_1) || owner->hasSpellbook() || owner->isCampaignYog()
					: !artifact->getWarMachine().hasValue() || !current->isWarMachineAvailable(id) || owner->hasArt(id)) continue;
				const int price = book ? GameConstants::SPELLBOOK_GOLD_COST : artifact->getPrice();
				auto shown = item("artifact", contextState.handle(owner->id.getNum()), id.getNum());
				shown["cost"] = cost(price);
				shown["locked"].Bool() = callback->getResourceAmount(EGameResID::GOLD) < price;
				append(result, "items", std::move(shown));
				if(callback->getResourceAmount(EGameResID::GOLD) < price) continue;
				auto selected = choice("buy_artifact", contextState.handle(owner->id.getNum()));
				selected["artifact"].Integer() = id.getNum(); selected["cost"] = cost(price);
				append(result, "choices", std::move(selected));
			}
		}
	}
	else if(contextState.panel == "visions")
	{
		const auto * caster = hero(*callback, contextState, subject);
		const CGObjectInstance * inspected = nullptr;
		for(const auto * target : visionsTargets(*callback, *caster))
		{
			auto selected = choice("inspect_object", subject);
			selected["destination"].String() = contextState.handle(target->id.getNum());
			selected["x"].Integer() = target->visitablePos().x; selected["y"].Integer() = target->visitablePos().y;
			append(result, "choices", std::move(selected));
			if(contextState.capabilityInspected == target->id.getNum()) inspected = target;
		}
		if(!inspected) contextState.capabilityInspected.reset();
		if(inspected)
		{
			const auto target = contextState.handle(inspected->id.getNum());
			result["inspected_subject"].String() = target;
			const auto statistic = [&](const char * kind, int id, int64_t amount)
			{
				auto shown = item(kind, target, id); shown["amount"].Integer() = amount; append(result, "items", std::move(shown));
			};
			const auto army = [&](const ArmyDescriptor & visible)
			{
				for(const auto & [slot, stack] : visible)
				{
					auto shown = item(visible.isDetailed ? "creature_exact" : "creature_quantity", target, stack.getId().getNum());
					shown["slot"].Integer() = slot.getNum(); shown["amount"].Integer() = stack.getCount();
					append(result, "items", std::move(shown));
				}
			};
			if(const auto * monster = dynamic_cast<const CGCreature *>(inspected))
			{
				const auto response = monster->getVisionsAction(caster);
				if(!response) throw std::runtime_error("Visions monster popup is no longer available");
				auto shown = item("creature_exact", target, monster->getCreatureID().getNum());
				shown["slot"].Integer() = 0; shown["amount"].Integer() = monster->getStackCount(SlotID(0));
				append(result, "items", std::move(shown));
				statistic("monster_intent", *response == CGCreature::FIGHT ? 0 : *response == CGCreature::FLEE ? 1 : *response == 0 ? 2 : 3,
					std::max(0, *response));
			}
			else if(inspected->ID == Obj::HERO)
			{
				InfoAboutHero info;
				if(!callback->getHeroInfo(inspected, info, caster)) throw std::runtime_error("Visions hero popup is no longer available");
				// This native view already applies Disguise. Never read the target
				// hero's actual army or private inventory to bypass that masking.
				army(info.army);
				if(info.details)
				{
					for(size_t id = 0; id < info.details->primskills.size(); ++id) statistic("primary_skill", id, info.details->primskills[id]);
					statistic("hero_stat", 0, info.details->mana); statistic("hero_stat", 1, info.details->morale); statistic("hero_stat", 2, info.details->luck);
				}
			}
			else
			{
				InfoAboutTown info;
				if(!callback->getTownInfo(inspected, info, caster)) throw std::runtime_error("Visions town popup is no longer available");
				army(info.army); statistic("town_stat", 2, info.fortLevel);
				if(info.details)
				{
					statistic("town_stat", 0, info.details->hallLevel); statistic("town_stat", 1, info.details->goldIncome);
					statistic("town_stat", 3, info.details->garrisonedHero ? 1 : 0);
				}
			}
		}
	}
	else if(contextState.panel == "artifacts")
	{
		std::vector<const CGHeroInstance *> owners{hero(*callback, contextState, subject)};
		if(contextState.capabilityParentContext["kind"].String() == "exchange")
		{
			const auto * other = hero(*callback, contextState, contextState.capabilityParentContext["selection"].String());
			if(!activeOwnHeroExchange(owners.front()->id.getNum(), other->id.getNum())) throw std::runtime_error("Artifact exchange query ended");
			owners.push_back(other);
		}
		for(const auto * source : owners) for(const auto sourceSlot : slots(*source))
		{
			const auto * artifact = source->getArt(sourceSlot, false);
			if(!artifact) continue;
			const auto * slot = source->getSlot(sourceSlot);
			auto shown = item("artifact", contextState.handle(source->id.getNum()), artifact->getTypeId().getNum());
			shown["slot"].Integer() = sourceSlot; shown["locked"].Bool() = slot->locked || vstd::contains(ArtifactUtils::unmovableSlots(), sourceSlot);
			if(artifact->isScroll()) shown["spell"].Integer() = artifact->getScrollSpellID().getNum();
			append(result, "items", std::move(shown));
			if(slot->locked || vstd::contains(ArtifactUtils::unmovableSlots(), sourceSlot)) continue;
			for(const auto * destination : owners)
			{
				auto destinations = slots(*destination);
				destinations.emplace_back(ArtifactPosition::BACKPACK_START + static_cast<int>(destination->artifactsInBackpack.size()));
				for(const auto destinationSlot : destinations)
				{
					if(vstd::contains(ArtifactUtils::unmovableSlots(), destinationSlot) || (source == destination && sourceSlot == destinationSlot)) continue;
					const bool backpack = ArtifactUtils::isSlotBackpack(destinationSlot);
					if(backpack && !explorer_rl::backpackInsertionFits(destination->artifactsInBackpack.size(),
						source == destination && ArtifactUtils::isSlotBackpack(sourceSlot),
						destinationSlot - ArtifactPosition::BACKPACK_START)) continue;
					const auto * occupied = destination->getArt(destinationSlot, false);
					const auto * info = destination->getSlot(destinationSlot);
					if(info && info->locked) continue;
					if(backpack ? artifact->getType()->isBig() || !ArtifactUtils::isBackpackFreeSlots(destination)
						: !artifact->canBePutAt(destination, destinationSlot, true) || (occupied && !occupied->canBePutAt(source, sourceSlot, true))) continue;
					auto selected = choice("move_artifact", contextState.handle(source->id.getNum()));
					selected["destination"].String() = contextState.handle(destination->id.getNum());
					selected["artifact"].Integer() = artifact->getTypeId().getNum();
					selected["slot"].Integer() = sourceSlot; selected["destination_slot"].Integer() = destinationSlot;
					append(result, "choices", std::move(selected));
				}
			}
			for(const auto * combined : ArtifactUtils::assemblyPossibilities(source, artifact->getTypeId()))
			{
				auto selected = choice("assemble_artifact", contextState.handle(source->id.getNum()));
				selected["slot"].Integer() = sourceSlot; selected["artifact"].Integer() = combined->getId().getNum();
				append(result, "choices", std::move(selected));
			}
			if(artifact->isCombined() && artifact->hasParts() && (!ArtifactUtils::isSlotBackpack(sourceSlot)
				|| ArtifactUtils::isBackpackFreeSlots(source, artifact->getType()->getConstituents().size() - 1)))
			{
				auto selected = choice("disassemble_artifact", contextState.handle(source->id.getNum()));
				selected["slot"].Integer() = sourceSlot; selected["artifact"].Integer() = artifact->getTypeId().getNum();
				append(result, "choices", std::move(selected));
			}
		}
	}
	else if(contextState.panel == "spellbook")
	{
		const auto * caster = hero(*callback, contextState, subject);
		for(const auto & spell : LIBRARY->spellh->objects)
		{
			if(!spell || !spell->isAdventure() || !caster->canCastThisSpell(spell.get())) continue;
			spells::detail::ProblemImpl problem;
			const bool allowed = spell->getAdventureMechanics().canBeCast(problem, callback.get(), caster);
			auto shown = item("spell", subject, spell->id.getNum());
			shown["amount"].Integer() = caster->getSpellCost(spell.get()); shown["locked"].Bool() = !allowed;
			append(result, "items", std::move(shown));
			if(!allowed) continue;
			const auto * ranged = spell->getAdventureMechanics().getEffectAs<AdventureSpellRangedEffect>(caster);
			if(!ranged)
			{
				auto selected = choice("cast_adventure_spell", subject); selected["spell"].Integer() = spell->id.getNum();
				append(result, "choices", std::move(selected));
			}
			else
			{
				const auto size = callback->getMapSize();
				for(int y = 0; y < size.y; ++y) for(int x = 0; x < size.x; ++x)
				{
					const int3 target(x, y, caster->visitablePos().z);
					if(!ranged->isTargetInRange(callback.get(), caster, target)) continue;
					if(callback->isVisible(target))
					{
						if(!spell->getAdventureMechanics().canBeCastAt(problem, callback.get(), caster, target)) continue;
					}
					else if(spell->getJsonKey() != "core:dimensionDoor") continue;
					// A fogged Dimension Door point is a public-range attempt.
					// Do not call native destination validation behind fog: its
					// occupancy test would reveal hidden objects through the mask.
					auto selected = choice("cast_adventure_spell", subject); selected["spell"].Integer() = spell->id.getNum();
					selected["x"].Integer() = x; selected["y"].Integer() = y;
					append(result, "choices", std::move(selected));
				}
			}
		}
	}
	else throw std::runtime_error("Unknown capability panel");
	return result;
}

void ExplorerRL::installCapabilityDialog()
{
	if(!expandedInterface || !contextState.capabilityDialog || contextState.capabilityDialog->installed) return;
	if(contextState.decision || contextState.executing) throw std::runtime_error("Capability query cannot replace an unfinished action");
	auto & pending = *contextState.capabilityDialog;
	pending.installed = true;
	contextState.capabilityParentContext = pending.panel["parent_context"];
	contextState.capabilityRef = pending.panel["ref"].String();
	JsonNode context;
	context["kind"].String() = "capability";
	context["subject"].String() = contextState.handle(pending.hero);
	context["panel"] = pending.panel["kind"];
	context["selection"] = pending.panel["ref"];
	change(contextState, context, callback->getDate(Date::DAY), "context_opened", -1, "capability_dialog");
}

void ExplorerRL::receiveLevelUp(const CGHeroInstance * current, PrimarySkill primary,
	const std::vector<SecondarySkill> & skills, QueryID query)
{
	try
	{
		if(!current || current->getOwner() != playerID) throw std::runtime_error("Level-up actor is not owned");
		std::lock_guard lock(stateMutex);
		const auto subject = contextState.handle(current->id.getNum());
		const auto ref = "capability-" + std::to_string(contextState.nextCapability++);
		auto payload = panel(ref, "level_up", mapContext(), query != QueryID(-1));
		payload["hero_level"].Integer() = current->level;
		payload["primary_skill"].Integer() = primary.getNum();
		State::CapabilityDialog pending{current->id.getNum(), query.getNum(), payload, {}};
		for(size_t index = 0; index < skills.size(); ++index)
		{
			auto selected = choice("choose_level_up", subject);
			selected["skill"].Integer() = skills[index].getNum();
			selected["skill_level"].Integer() = current->getSecSkillLevel(skills[index]) + 1;
			append(pending.panel, "choices", std::move(selected));
			pending.replies.emplace(pending.panel["choices"].Vector().back()["ref"].String(), static_cast<int>(index));
		}
		if(query == QueryID(-1))
		{
			// The native engine already applied zero/one-choice level-ups. Show
			// that delivered outcome without manufacturing a strategic decision.
			pending.replies.clear();
		}
		if(query != QueryID(-1) && contextState.capabilityDialog) throw std::runtime_error("Overlapping native capability queries");
		if(!contextState.executing) contextState.supersede(callback->getDate(Date::DAY), "capability_query");
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = query == QueryID(-1) ? "level_gained" : "capability_requested";
		record["event"]["subject"].String() = subject;
		record["event"]["capability"] = pending.panel;
		contextState.records.push_back(std::move(record));
		if(query != QueryID(-1)) contextState.capabilityDialog = std::move(pending);
		++contextState.interactions;
		changed.notify_all();
	}
	catch(const std::exception &) { unsupported("unsupported_interaction: native level-up"); }
}

void ExplorerRL::showMapObjectSelectDialog(QueryID query, const Component & icon, const MetaString &,
	const MetaString &, const std::vector<ObjectInstanceID> & objects)
{
	if(!expandedInterface) { unsupported("object selection"); return; }
	try
	{
		std::lock_guard lock(stateMutex);
		if(icon.type != ComponentType::SPELL || query == QueryID(-1) || !contextState.executingHero || contextState.capabilityDialog)
			throw std::runtime_error("Unsupported object-selection origin");
		const auto spell = icon.subType.as<SpellID>();
		if(spell != SpellID::TOWN_PORTAL) throw std::runtime_error("Unsupported object-selection spell");
		const auto ref = "capability-" + std::to_string(contextState.nextCapability++);
		const auto subject = contextState.handle(*contextState.executingHero);
		State::CapabilityDialog pending{*contextState.executingHero, query.getNum(), panel(ref, "spell_target", mapContext(), true), {}};
		for(const auto id : objects)
		{
			const auto * destination = callback->getTown(id);
			if(!destination || destination->getOwner() != playerID) throw std::runtime_error("Unowned spell destination");
			auto selected = choice("choose_spell_target", subject);
			selected["destination"].String() = contextState.handle(id.getNum()); selected["spell"].Integer() = spell.getNum();
			append(pending.panel, "choices", std::move(selected));
			pending.replies.emplace(pending.panel["choices"].Vector().back()["ref"].String(), id.getNum());
		}
		auto cancel = choice("cancel_spell_target", subject); cancel["spell"].Integer() = spell.getNum();
		append(pending.panel, "choices", std::move(cancel));
		pending.replies.emplace(pending.panel["choices"].Vector().back()["ref"].String(), -1);
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = "capability_requested";
		record["event"]["subject"].String() = subject;
		record["event"]["capability"] = pending.panel;
		contextState.records.push_back(std::move(record));
		contextState.capabilityDialog = std::move(pending);
		++contextState.interactions;
		changed.notify_all();
	}
	catch(const std::exception &) { unsupported("unsupported_interaction: spell target"); }
}

void ExplorerRL::showWorldViewEx(const std::vector<ObjectPosInfo> & objects, bool showTerrain)
{
	if(!expandedInterface) { if(contextual) unsupported("unsupported_interaction: world view"); return; }
	try
	{
		std::lock_guard lock(stateMutex);
		if(!contextState.executingHero || contextState.capabilityDialog) throw std::runtime_error("World view without its cast");
		const auto subject = contextState.handle(*contextState.executingHero);
		const auto ref = "capability-" + std::to_string(contextState.nextCapability++);
		State::CapabilityDialog pending{*contextState.executingHero, -1, panel(ref, "world_view", mapContext()), {}};
		for(const auto & object : objects)
		{
			auto shown = item("world_object", subject, static_cast<int>(object.id));
			shown["amount"].Integer() = object.subId;
			shown["owner"].String() = object.owner == playerID ? "self" : object.owner.isValidPlayer() ? "opponent" : "neutral";
			shown["x"].Integer() = object.pos.x; shown["y"].Integer() = object.pos.y; shown["z"].Integer() = object.pos.z;
			append(pending.panel, "items", std::move(shown));
		}
		if(showTerrain)
		{
			// This delivered spell window explicitly reveals terrain. The
			// unrestricted tile read is confined to its terrain ID, never objects,
			// passability, armies, or occupancy behind fog of war.
			const auto size = callback->getMapSize();
			for(int z = 0; z < size.z; ++z) for(int y = 0; y < size.y; ++y) for(int x = 0; x < size.x; ++x)
			{
				const auto * tile = callback->getTileUnchecked(int3(x, y, z));
				if(!tile) throw std::runtime_error("Missing revealed terrain tile");
				auto shown = item("world_terrain", subject, tile->getTerrainID().getNum());
				shown["x"].Integer() = x; shown["y"].Integer() = y; shown["z"].Integer() = z;
				append(pending.panel, "items", std::move(shown));
			}
		}
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = "capability_requested";
		record["event"]["subject"].String() = subject;
		record["event"]["capability"] = pending.panel;
		contextState.records.push_back(std::move(record));
		contextState.capabilityDialog = std::move(pending);
		++contextState.interactions;
		changed.notify_all();
	}
	catch(const std::exception &) { unsupported("unsupported_interaction: revealed world view"); }
}

bool ExplorerRL::executeCapabilityAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool opening = kind == "open_capability", closing = kind == "close_capability";
	if(!expandedInterface || !action["subject"].isString() || !action["x"].isNull() || !action["y"].isNull() || !action["destination"].isNull()
		|| (opening ? !action["panel"].isString() : !action["panel"].isNull())
		|| ((opening || closing) ? !action["selection"].isNull() : !action["selection"].isString())
		|| (kind == "trade_resource" ? action["count"].getType() != JsonNode::JsonType::DATA_INTEGER : !action["count"].isNull()))
		throw std::runtime_error("Incorrect typed capability action parameters");
	const auto consume = [&](const char * consumer)
	{
		contextState.executing = true; contextState.commands = 0; contextState.routeInteractions.reset();
		auto record = contextState.record("consumed", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision; record["consumer"].String() = consumer;
		contextState.records.push_back(std::move(record));
	};
	const auto finish = [&](const char * status, bool accepted)
	{
		JsonNode result; result["status"].String() = status; result["accepted"].Bool() = accepted;
		for(const char * key : {"position", "movement", "destination_reached"}) result[key] = JsonNode();
		contextualResult(std::move(result), decision);
	};
	if(opening || closing)
	{
		{
			const auto gameLock = readGameState(); captureContextual(); std::lock_guard lock(stateMutex);
			if(contextState.decision != decision || terminal || stopping) return false;
			if(contextState.subject != action["subject"].String()) throw std::runtime_error("Capability navigation subject mismatch");
			JsonNode next;
			if(opening)
			{
				const auto controls = capabilityControls();
				if(std::find(controls.Vector().begin(), controls.Vector().end(), action["panel"]) == controls.Vector().end())
					throw std::runtime_error("Capability control was not displayed");
				contextState.capabilityParentContext = contextState.contextJson();
				contextState.capabilityInspected.reset();
				contextState.capabilityRef = "capability-" + std::to_string(contextState.nextCapability++);
				next["kind"].String() = "capability"; next["subject"] = action["subject"]; next["panel"] = action["panel"]; next["selection"] = JsonNode();
			}
			else
			{
				if(contextState.context != "capability" || capabilityPanel()["mandatory"].Bool()) throw std::runtime_error("Mandatory capability requires a native choice");
				next = contextState.capabilityParentContext;
				contextState.capabilityDialog.reset();
			}
			consume("adapter");
			change(contextState, next, callback->getDate(Date::DAY), "context", decision);
		}
		finish(opening ? "opened" : "closed", true); return true;
	}
	if(kind == "inspect_object")
	{
		{
			const auto gameLock = readGameState(); captureContextual(); std::lock_guard lock(stateMutex);
			if(contextState.decision != decision || terminal || stopping) return false;
			if(contextState.context != "capability" || contextState.panel != "visions" || contextState.subject != action["subject"].String())
				throw std::runtime_error("Visions inspection context mismatch");
			const auto current = capabilityPanel();
			const auto & choices = current["choices"].Vector();
			const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto & value) { return value["kind"] == action["kind"] && value["ref"] == action["selection"]; });
			if(found == choices.end()) throw std::runtime_error("Visions target was not displayed");
			consume("adapter");
			contextState.capabilityInspected = resolve(contextState, (*found)["destination"].String());
		}
		finish("inspected", true); return true;
	}
	JsonNode selected;
	int query = -1, reply = -1;
	const bool cast = kind == "cast_adventure_spell";
	const bool answer = kind == "choose_level_up" || kind == "choose_spell_target" || kind == "cancel_spell_target";
	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto * actor = kind == "trade_resource" ? nullptr : hero(*callback, contextState, selected["subject"].String());
		if(kind == "trade_resource")
		{
			const auto * market = town(*callback, contextState, selected["subject"].String());
			callback->trade(market->id, EMarketMode::RESOURCE_RESOURCE,
				GameResID(selected["resource_from"].Integer()), GameResID(selected["resource_to"].Integer()),
				static_cast<ui32>(action["count"].Integer() * selected["give"].Integer()));
		}
		else if(kind == "buy_artifact") callback->buyArtifact(actor, ArtifactID(selected["artifact"].Integer()));
		else if(kind == "move_artifact")
		{
			const auto * destination = hero(*callback, contextState, selected["destination"].String());
			const ArtifactPosition sourceSlot(selected["slot"].Integer());
			const ArtifactPosition destinationSlot(selected["destination_slot"].Integer());
			if(ArtifactUtils::isSlotBackpack(destinationSlot)
				&& !explorer_rl::backpackInsertionFits(destination->artifactsInBackpack.size(),
					actor == destination && ArtifactUtils::isSlotBackpack(sourceSlot),
					destinationSlot - ArtifactPosition::BACKPACK_START))
				throw std::runtime_error("Artifact destination exceeds the backpack after source removal");
			if(!callback->swapArtifacts(ArtifactLocation(actor->id, sourceSlot),
				ArtifactLocation(destination->id, destinationSlot)))
				throw std::runtime_error("Displayed artifact operation failed to issue");
		}
		else if(kind == "assemble_artifact" || kind == "disassemble_artifact")
			callback->assembleArtifacts(actor->id, ArtifactPosition(selected["slot"].Integer()), kind == "assemble_artifact", ArtifactID(selected["artifact"].Integer()));
		else if(cast)
		{
			const int3 target = selected["x"].isNull() ? int3(-1, -1, -1)
				: int3(selected["x"].Integer(), selected["y"].Integer(), actor->visitablePos().z);
			callback->castSpell(actor, SpellID(selected["spell"].Integer()), target);
		}
		else if(answer) callback->sendQueryReply(kind == "cancel_spell_target" ? std::nullopt : std::optional<int32_t>(reply), QueryID(query));
		else throw std::runtime_error("Unknown capability mutation");
	}, false, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		if(contextState.context != "capability" || contextState.subject != action["subject"].String()) throw std::runtime_error("Capability action context mismatch");
		const auto current = capabilityPanel();
		const auto & choices = current["choices"].Vector();
		const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto & value) { return value["ref"] == action["selection"] && value["kind"] == action["kind"]; });
		if(found == choices.end()) throw std::runtime_error("Typed capability choice was not displayed");
		selected = *found;
		if(kind == "trade_resource" && (action["count"].Integer() < 1 || action["count"].Integer() > selected["max_count"].Integer()))
			throw std::runtime_error("Trade lot count is outside displayed bounds");
		if(answer)
		{
			if(!contextState.capabilityDialog || !contextState.capabilityDialog->installed) throw std::runtime_error("No native capability query");
			query = contextState.capabilityDialog->query;
			reply = contextState.capabilityDialog->replies.at(action["selection"].String());
		}
		consume("engine"); contextState.issueArmed = true;
		if(cast || answer)
		{
			contextState.executingHero = resolve(contextState, selected["subject"].String());
			contextState.capabilityDialog.reset();
			change(contextState, mapContext(), callback->getDate(Date::DAY), "context", decision);
		}
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	if(answer && !accepted) throw std::runtime_error("Engine rejected the mandatory capability reply");
	finish(accepted ? "capability_applied" : "rejected", accepted);
	return true;
}

void ExplorerRL::receiveGarrison(const CArmedInstance * joining, const CGHeroInstance * owner, bool removable, QueryID query)
{
	try
	{
		if(!joining || !owner || owner->getOwner() != playerID || query == QueryID(-1)
			|| !callback->getObj(joining->id, false)) throw std::runtime_error("Unsupported garrison query");
		const bool neutral = joining->ID == Obj::MONSTER && !joining->getOwner().isValidPlayer() && removable;
		const bool mine = joining->ID == Obj::MINE;
		const bool garrison = joining->ID == Obj::GARRISON || joining->ID == Obj::GARRISON2;
		const bool self = joining->getOwner() == playerID;
		const bool allied = joining->getOwner().isValidPlayer()
			&& callback->getPlayerRelations(playerID, joining->getOwner()) == PlayerRelations::ALLIES;
		if(!neutral && !((mine && self && removable) || (garrison && (self || allied))))
			throw std::runtime_error("Unsupported garrison participants");
		std::lock_guard lock(stateMutex);
		if(contextState.ownHeroExchange || contextState.adventureDialog || contextState.capabilityDialog
			|| contextState.context != "map" || !contextState.executing || !contextState.decision)
			throw std::runtime_error("Unexpected garrison-dialog lifecycle");
		contextState.ownHeroExchange = State::OwnHeroExchange{
			owner->id.getNum(), joining->id.getNum(), query.getNum(), false, neutral, !neutral, removable, allied};
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = neutral ? "army_join_requested"
			: removable ? "garrison_requested" : "garrison_locked_requested";
		record["event"]["subject"].String() = contextState.handle(owner->id.getNum());
		record["event"]["partner"].String() = contextState.handle(joining->id.getNum());
		if(!neutral)
		{
			record["event"]["object_kind"].Integer() = joining->ID.getNum();
			record["event"]["owner"].String() = allied ? "ally" : "self";
		}
		contextState.records.push_back(std::move(record));
		++contextState.interactions; changed.notify_all();
	}
	catch(const std::exception & error)
	{
		// Report the delivered visible window type, never native query IDs,
		// private armies or arbitrary game text. Do not call every garrison
		// error a neutral join: mines and map garrisons share this callback.
		unsupported("unsupported_interaction: garrison dialog: " + std::string(error.what())
			+ "; object_kind=" + (joining ? std::to_string(joining->ID.getNum()) : "missing")
			+ "; removable=" + (removable ? "true" : "false"));
	}
}
