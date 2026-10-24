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
