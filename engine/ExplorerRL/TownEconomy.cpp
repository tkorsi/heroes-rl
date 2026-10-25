#include "StdInc.h"
#include "TownEconomy.h"

#include "../../lib/CCreatureHandler.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/entities/building/CBuilding.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"

#include <regex>
#include <limits>
#include <type_traits>

namespace explorer_rl
{
namespace
{
[[noreturn]] void unsupported()
{
	throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: town economy data");
}

std::string buildingText(const CBuilding & building, bool description = false)
{
	const auto id = description ? building.getDescriptionTextID() : building.getNameTextID();
	static const std::regex name("building\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name");
	static const std::regex detail("building\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.description");
	if(!std::regex_match(id, description ? detail : name))
		unsupported();
	return id;
}

const char * statusName(EBuildingState status)
{
	switch(status)
	{
	case EBuildingState::HAVE_CAPITAL: return "have_capital";
	case EBuildingState::NO_WATER: return "no_water";
	case EBuildingState::FORBIDDEN: return "forbidden";
	case EBuildingState::ADD_MAGES_GUILD: return "add_mages_guild";
	case EBuildingState::ALREADY_PRESENT: return "already_present";
	case EBuildingState::CANT_BUILD_TODAY: return "cant_build_today";
	case EBuildingState::NO_RESOURCES: return "no_resources";
	case EBuildingState::ALLOWED: return "allowed";
	case EBuildingState::PREREQUIRES: return "prerequires";
	case EBuildingState::MISSING_BASE: return "missing_base";
	default: unsupported();
	}
}

JsonNode requirement(const CGTownInstance & town, const CBuilding::TRequired::Variant & expression,
	size_t & remaining, int depth = 0)
{
	if(remaining == 0 || depth > 16)
		unsupported();
	--remaining;
	return std::visit([&](const auto & item)
	{
		using Item = std::decay_t<decltype(item)>;
		JsonNode node;
		node["name_text_id"] = JsonNode();
		node["children"].Vector();
		if constexpr(std::is_same_v<Item, BuildingID>)
		{
			const auto found = town.getTown()->buildings.find(item);
			if(found == town.getTown()->buildings.end()) unsupported();
			node["op"].String() = "building";
			node["name_text_id"].String() = buildingText(*found->second);
		}
		else
		{
			if constexpr(std::is_same_v<Item, CBuilding::TRequired::OperatorAll>) node["op"].String() = "all";
			else if constexpr(std::is_same_v<Item, CBuilding::TRequired::OperatorAny>) node["op"].String() = "any";
			else node["op"].String() = "none";
			for(const auto & child : item.expressions)
				node["children"].Vector().push_back(requirement(town, child, remaining, depth + 1));
		}
		return node;
	}, expression);
}
}

std::string TownEconomyState::buildingHandle(const CGTownInstance & town, BuildingID building)
{
	const auto key = std::make_pair(town.id.getNum(), building.getNum());
	if(const auto found = buildingHandles.find(key); found != buildingHandles.end())
		return found->second;
	const auto ref = "building-" + std::to_string(buildingHandles.size());
	buildingHandles.emplace(key, ref);
	buildings.emplace(ref, key);
	return ref;
}

BuildingID TownEconomyState::resolveBuilding(const CGTownInstance & town, const std::string & ref) const
{
	const auto found = buildings.find(ref);
	if(found == buildings.end() || found->second.first != town.id.getNum())
		throw std::runtime_error("Building reference does not belong to the active town");
	return BuildingID(found->second.second);
}

std::string TownEconomyState::dwellingHandle(const CGTownInstance & town, int level)
{
	const auto key = std::make_pair(town.id.getNum(), level);
	if(const auto found = dwellingHandles.find(key); found != dwellingHandles.end()) return found->second;
	const auto ref = "dwelling-" + std::to_string(dwellingHandles.size());
	dwellingHandles.emplace(key, ref);
	dwellings.emplace(ref, key);
	return ref;
}

int TownEconomyState::resolveDwelling(const CGTownInstance & town, const std::string & ref) const
{
	const auto found = dwellings.find(ref);
	if(found == dwellings.end() || found->second.first != town.id.getNum())
		throw std::runtime_error("Dwelling reference does not belong to the active town");
	const int level = found->second.second;
	if(level < 0 || static_cast<size_t>(level) >= town.creatures.size() || town.creatures[level].second.empty())
		throw std::runtime_error("Dwelling control is no longer present");
	return level;
}

std::string TownEconomyState::recruitHandle(const CGTownInstance & town, int level, CreatureID creature)
{
	const auto key = std::make_tuple(town.id.getNum(), level, creature.getNum());
	if(const auto found = recruitHandles.find(key); found != recruitHandles.end()) return found->second;
	const auto ref = "recruit-" + std::to_string(recruitHandles.size());
	recruitHandles.emplace(key, ref);
	return ref;
}

std::string TownEconomyState::destinationHandle(const CGTownInstance & town)
{
	const auto * destination = town.getUpperArmy();
	if(!destination || destination->getOwner() != town.getOwner()) unsupported();
	const auto key = std::make_pair(town.id.getNum(), destination->id.getNum());
	if(const auto found = destinationHandles.find(key); found != destinationHandles.end()) return found->second;
	const auto ref = "army-" + std::to_string(destinationHandles.size());
	destinationHandles.emplace(key, ref);
	return ref;
}

JsonNode recruitmentCapacityNotice()
{
	// GUIClasses.cpp CRecruitmentWindow::buy uses allTexts[17] and the
	// ordinary modal CInfoWindow, with no components or server query.
	JsonNode event;
	event["kind"].String() = "notice";
	auto & notice = event["notice"];
	notice["display_mode"].String() = "modal";
	notice["components"].Vector();
	JsonNode token;
	token["op"].String() = "append_text";
	token["text_id"].String() = "core.genrltxt.17";
	token["number"] = JsonNode();
	token["owner"] = JsonNode();
	notice["text"].Vector().push_back(std::move(token));
	return event;
}

std::vector<const CBuilding *> displayedHallBuildings(const CGTownInstance & town)
{
	std::vector<const CBuilding *> result;
	for(const auto & row : town.getTown()->clientInfo.hallSlots)
		for(const auto & slot : row)
		{
			const CBuilding * displayed = nullptr;
			for(const auto id : slot)
			{
				if(!id.hasValue()) continue; // same invalid-entry skip as native hall
				const auto found = town.getTown()->buildings.find(id);
				if(found == town.getTown()->buildings.end()) unsupported();
				const auto * current = found->second.get();
				if(town.hasBuilt(id)) displayed = current;
				else if(current->mode == CBuilding::BUILD_NORMAL)
				{
					displayed = current;
					break;
				}
			}
			if(displayed) result.push_back(displayed);
		}
	if(result.size() > 128) unsupported();
	return result;
}

bool hallHasGrailPrompt(const CGTownInstance & town)
{
	const auto * hero = town.getVisitingHero();
	if(hero && hero->getOwner() != town.getOwner()) unsupported();
	return hero && hero->hasArt(ArtifactID::GRAIL)
		&& !town.hasBuilt(BuildingID::GRAIL);
}

JsonNode townPanel(const CCallback & player, const CGTownInstance & town,
	const std::string & panel, const std::string & selection, TownEconomyState & state)
{
	if(player.getPlayerID() != town.getOwner())
		throw std::runtime_error("Town panel requires player ownership");
	JsonNode result;
	result["kind"].String() = panel;
	result["controls"].Vector();
	result["buildings"].Vector();
	result["building"] = JsonNode();
	if(panel == "overview")
	{
		JsonNode control;
		control["panel"].String() = "hall";
		control["ref"].String() = "panel-0";
		result["controls"].Vector().push_back(std::move(control));
		// CCastleBuildings/CCreaInfo offer the popup for every nonempty
		// dwelling list, including an exhausted stock. No stock or cost is read.
		for(size_t level = 0; level < town.creatures.size(); ++level)
			if(!town.creatures[level].second.empty())
			{
				JsonNode dwelling;
				dwelling["panel"].String() = "recruitment";
				dwelling["ref"].String() = state.dwellingHandle(town, static_cast<int>(level));
				result["controls"].Vector().push_back(std::move(dwelling));
			}
	}
	else if(panel == "hall")
	{
		for(const auto * building : displayedHallBuildings(town))
		{
			JsonNode choice;
			choice["ref"].String() = state.buildingHandle(town, building->bid);
			choice["name_text_id"].String() = buildingText(*building);
			choice["status"].String() = statusName(player.canBuildStructure(&town, building->bid));
			result["buildings"].Vector().push_back(std::move(choice));
		}
	}
	else if(panel == "building")
	{
		const auto id = state.resolveBuilding(town, selection);
		const auto & building = *town.getTown()->buildings.at(id);
		const auto status = player.canBuildStructure(&town, id);
		auto & detail = result["building"];
		detail["ref"].String() = selection;
		detail["name_text_id"].String() = buildingText(building);
		detail["description_text_id"].String() = buildingText(building, true);
		detail["status"].String() = statusName(status);
		detail["cost"].Vector();
		for(int resource = 0; resource < 7; ++resource)
		{
			if(building.resources[resource] < 0) unsupported();
			detail["cost"].Vector().emplace_back(building.resources[resource]);
		}
		for(size_t resource = 7; resource < building.resources.size(); ++resource)
			if(building.resources[resource] != 0) unsupported();
		detail["requirements"] = JsonNode();
		size_t remaining = 128;
		if(status == EBuildingState::PREREQUIRES)
			detail["requirements"] = requirement(town, town.genBuildingRequirements(id).get(), remaining);
		else if(status == EBuildingState::MISSING_BASE)
			detail["requirements"] = requirement(town, CBuilding::TRequired::Variant(building.upgrade), remaining);
	}
	else if(panel == "recruitment")
	{
		const int level = state.resolveDwelling(town, selection);
		const auto & pool = town.creatures[level];
		if(pool.first > std::numeric_limits<int32_t>::max()
			|| pool.second.size() > 32 || state.selectedRecruitIndex >= pool.second.size()) unsupported();
		auto & recruit = result["recruitment"];
		recruit["control_ref"].String() = selection;
		recruit["stock"].Integer() = pool.first;
		// Native availableCreaturesChanged reverses the variants and initially
		// selects card zero; refresh preserves its card index, not a hidden ID.
		for(auto creature = pool.second.rbegin(); creature != pool.second.rend(); ++creature)
		{
			const auto * type = creature->toCreature();
			if(!type || creature->getNum() < 0 || type->warMachine.hasValue()) unsupported();
			JsonNode card;
			card["ref"].String() = state.recruitHandle(town, level, *creature);
			card["creature"].Integer() = creature->getNum();
			recruit["offers"].Vector().push_back(std::move(card));
		}
		const auto creature = pool.second[pool.second.size() - 1 - state.selectedRecruitIndex];
		recruit["selected_offer_ref"].String() = state.recruitHandle(town, level, creature);
		const auto & cost = creature.toCreature()->getFullRecruitCost();
		for(int resource = 0; resource < 7; ++resource)
		{
			if(cost[resource] < 0) unsupported();
			recruit["unit_cost"].Vector().emplace_back(cost[resource]);
		}
		for(size_t resource = 7; resource < cost.size(); ++resource)
			if(cost[resource] != 0) unsupported();
		const auto maximum = std::min<int64_t>(pool.first, creature.toCreature()->maxAmount(player.getResourceAmount()));
		if(maximum < 0) unsupported();
		recruit["slider_max"].Integer() = maximum;
		recruit["destination"]["ref"].String() = state.destinationHandle(town);
		recruit["destination"]["role"].String() = town.getGarrisonHero() ? "garrison_hero" : "town_garrison";
	}
	else unsupported();
	return result;
}
}
