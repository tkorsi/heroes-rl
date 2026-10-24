#include "StdInc.h"
#include "TownEconomy.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/entities/building/CBuilding.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"

#include <regex>
#include <type_traits>

namespace explorer_rl
{
namespace
{
[[noreturn]] void unsupported()
{
	throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: town construction data");
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
	else unsupported();
	return result;
}
}
