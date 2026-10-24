#pragma once

#include "../../lib/constants/EntityIdentifiers.h"
#include "../../lib/json/JsonNode.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class CBuilding;
class CCallback;
class CGTownInstance;
VCMI_LIB_NAMESPACE_END

namespace explorer_rl
{
/// Internal routing only. Handles are assigned when native controls or cards
/// are first displayed; instance IDs never enter the player wire.
struct TownEconomyState
{
	std::map<std::pair<int, int>, std::string> buildingHandles;
	std::map<std::string, std::pair<int, int>> buildings;
	std::map<std::pair<int, int>, std::string> dwellingHandles;
	std::map<std::string, std::pair<int, int>> dwellings;
	std::map<std::tuple<int, int, int>, std::string> recruitHandles;
	std::map<std::pair<int, int>, std::string> destinationHandles;
	size_t selectedRecruitIndex = 0;
	std::string buildingHandle(const CGTownInstance & town, BuildingID building);
	BuildingID resolveBuilding(const CGTownInstance & town, const std::string & ref) const;
	std::string dwellingHandle(const CGTownInstance & town, int level);
	int resolveDwelling(const CGTownInstance & town, const std::string & ref) const;
	std::string recruitHandle(const CGTownInstance & town, int level, CreatureID creature);
	std::string destinationHandle(const CGTownInstance & town);
};

/// The same single building per hall slot chosen by CHallInterface. This is
/// not the faction's whole building catalog or the remaining upgrade chain.
std::vector<const CBuilding *> displayedHallBuildings(const CGTownInstance & town);

/// Only the active panel's displayed fields. Caller holds a stable game-state
/// read view and proves player ownership before calling. Overlays never read
/// underlying armies; recruitment stocks/costs are read only in that popup.
JsonNode townPanel(const CCallback & player, const CGTownInstance & town,
	const std::string & panel, const std::string & selection, TownEconomyState & state);

/// Called only on explicit hall entry, where native UI tests this condition
/// before opening CHallInterface. It does not disclose the artifact inventory.
bool hallHasGrailPrompt(const CGTownInstance & town);

/// Exact no-slot information shown by CRecruitmentWindow::buy for a town.
/// Called only by the v3 adapter's matching, consumed no-capacity branch.
JsonNode recruitmentCapacityNotice();
}
