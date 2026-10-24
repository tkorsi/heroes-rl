#pragma once

#include "../../lib/constants/EntityIdentifiers.h"
#include "../../lib/json/JsonNode.h"

#include <map>
#include <string>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class CBuilding;
class CCallback;
class CGTownInstance;
VCMI_LIB_NAMESPACE_END

namespace explorer_rl
{
/// Internal routing only. Handles are assigned when a native hall entry is
/// first displayed; numeric town/building IDs never enter the player wire.
struct TownEconomyState
{
	std::map<std::pair<int, int>, std::string> buildingHandles;
	std::map<std::string, std::pair<int, int>> buildings;
	std::string buildingHandle(const CGTownInstance & town, BuildingID building);
	BuildingID resolveBuilding(const CGTownInstance & town, const std::string & ref) const;
};

/// The same single building per hall slot chosen by CHallInterface. This is
/// not the faction's whole building catalog or the remaining upgrade chain.
std::vector<const CBuilding *> displayedHallBuildings(const CGTownInstance & town);

/// Only the active panel's displayed fields. Caller holds a stable game-state
/// read view and proves player ownership before calling; overlays do not read
/// underlying armies, hero inventories, recruitment stocks or spell lists.
JsonNode townPanel(const CCallback & player, const CGTownInstance & town,
	const std::string & panel, const std::string & selection, TownEconomyState & state);

/// Called only on explicit hall entry, where native UI tests this condition
/// before opening CHallInterface. It does not disclose the artifact inventory.
bool hallHasGrailPrompt(const CGTownInstance & town);
}
