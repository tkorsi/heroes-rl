#pragma once

#include "../../lib/json/JsonNode.h"
#include "TownEconomy.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace explorer_rl
{
/// Per-player adapter bookkeeping. All access is protected by ExplorerRL's
/// state mutex; engine identifiers never leave these routing tables.
struct ContextualState
{
	std::map<int, std::string> handles;
	std::map<std::string, int> subjects;
	std::vector<JsonNode> records;
	JsonNode lastObservation;
	int lastDay = 0;
	std::string context = "map";
	std::string subject;
	std::string panel; ///< empty outside the opt-in v3 town context
	std::string selection;
	TownEconomyState economy;
	std::map<int, std::string> tavernHandles; ///< private offered HeroTypeID routing
	size_t selectedTavernIndex = 0;
	std::map<std::pair<int, int>, std::string> armyHandles;
	std::map<std::tuple<int, int, int>, std::string> armySlotHandles;
	JsonNode stackParentContext; ///< exact opened parent; never refreshed while stack is open
	std::string selectedUpgradeRef;
	std::map<int, std::string> upgradeHandles; ///< private CreatureID routing
	struct OwnHeroExchange
	{
		int first;
		int second;
		int query;
		bool installed = false;
		bool joiningNeutral = false;
		bool objectGarrison = false;
		bool removableUnits = true;
		bool allied = false;
	};
	std::optional<OwnHeroExchange> ownHeroExchange; ///< private native query, never policy input
	struct AdventureDialog
	{
		int hero;
		int query;
		JsonNode panel;
		std::map<std::string, int> replies;
		bool installed = false;
	};
	std::optional<AdventureDialog> adventureDialog; ///< private query/reply routing
	struct CapabilityDialog
	{
		int hero;
		int query;
		JsonNode panel;
		std::map<std::string, int> replies;
		bool installed = false;
	};
	std::optional<CapabilityDialog> capabilityDialog;
	JsonNode capabilityParentContext;
	std::string capabilityRef;
	std::optional<int> capabilityInspected;
	uint64_t nextCapability = 0;
	std::optional<int> executingHero; ///< own subject of current move or dialog answer
	uint64_t nextDialog = 0;
	int64_t revision = -1;
	int64_t nextDecision = 0;
	std::optional<int64_t> decision;
	bool executing = false;
	bool issueArmed = false;
	int64_t commands = 0;
	int64_t interactions = 0;
	std::optional<int64_t> routeInteractions;
	struct Request { int64_t decision; int64_t command; };
	std::map<int, Request> requests;

	std::string handle(int engineID);
	JsonNode contextJson() const;
	JsonNode record(const char * kind, int day) const;
	void supersede(int day, const char * reason);
};
}
