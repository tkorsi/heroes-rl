#include "StdInc.h"
#include "ExplorerRL.h"

#include "../../lib/CCreatureHandler.h"
#include "../../lib/IGameSettings.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"

#include <limits>

namespace
{
using ArmySnapshot = std::map<int, std::pair<int, int64_t>>;

ArmySnapshot recruitmentArmy(const CArmedInstance & army)
{
	ArmySnapshot result;
	for(const auto & [slot, stack] : army.Slots())
		result.emplace(slot.getNum(), std::make_pair(stack->getCreatureID().getNum(), stack->getCount()));
	return result;
}

JsonNode recruitmentResult(const char * status, bool accepted)
{
	JsonNode result;
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	for(const char * field : {"position", "movement", "destination_reached"}) result[field] = JsonNode();
	return result;
}
}

bool ExplorerRL::executeRecruitmentAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool buying = kind == "recruit";
	const bool opening = kind == "open_panel";
	if(!action["subject"].isString() || !action["x"].isNull() || !action["y"].isNull()
		|| (opening ? action["panel"].String() != "recruitment" : !action["panel"].isNull())
		|| (kind == "back" ? !action["selection"].isNull() : !action["selection"].isString()))
		throw std::runtime_error("Incorrect recruitment action parameters");
	if(buying)
	{
		if(action["count"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| action["count"].Integer() < 1 || action["count"].Integer() > std::numeric_limits<int32_t>::max()
			|| !action["destination"].isString())
			throw std::runtime_error("Recruitment requires an explicit positive count and destination");
	}
	else if(!action["count"].isNull() || !action["destination"].isNull())
		throw std::runtime_error("Unused recruitment action parameter");
	const auto subject = action["subject"].String();
	const auto selection = action["selection"].isString() ? action["selection"].String() : std::string();
	const int64_t count = buying ? action["count"].Integer() : 0;
	ObjectInstanceID townID(-1), destinationID(-1);
	CreatureID creatureID(-1);
	int level = -1;
	SlotID slot;
	std::array<int64_t, 7> resources{}, debit{};
	CGDwelling::TCreaturesSet beforePools;
	ArmySnapshot beforeArmy;
	bool noCapacity = false;

	// As for construction, all helpers run with the engine read lock and
	// stateMutex held. The popup never offers a visiting hero as destination.
	const auto activeTown = [&]() -> const CGTownInstance *
	{
		if(contextState.context != "town" || contextState.subject != subject)
			throw std::runtime_error("Recruitment does not match the active town context");
		const auto handle = contextState.subjects.find(subject);
		if(handle == contextState.subjects.end()) throw std::runtime_error("Unknown recruitment town");
		for(const auto * town : callback->getTownsInfo())
			if(town->id.getNum() == handle->second && town->getOwner() == playerID) return town;
		throw std::runtime_error("Recruitment town is no longer owned");
	};
	const auto validate = [&](const CGTownInstance & town)
	{
		if(opening)
		{
			if(contextState.panel != "overview") throw std::runtime_error("Recruitment entry requires town overview");
			const auto & controls = contextState.lastObservation["window"]["town_panel"]["controls"].Vector();
			if(std::none_of(controls.begin(), controls.end(), [&](const auto & control)
				{ return control["panel"].String() == "recruitment" && control["ref"].String() == selection; }))
				throw std::runtime_error("Recruitment control was not displayed");
			contextState.economy.resolveDwelling(town, selection);
			return;
		}
		if(contextState.panel != "recruitment") throw std::runtime_error("Recruitment requires its open popup");
		if(kind == "back") return;
		const auto & panel = contextState.lastObservation["window"]["town_panel"]["recruitment"];
		const auto & offers = panel["offers"].Vector();
		if(std::none_of(offers.begin(), offers.end(), [&](const auto & offer) { return offer["ref"].String() == selection; }))
			throw std::runtime_error("Recruitment card was not displayed");
		if(buying && (panel["selected_offer_ref"].String() != selection
			|| action["destination"] != panel["destination"]["ref"] || count > panel["slider_max"].Integer()))
			throw std::runtime_error("Recruitment selection, count or destination is unavailable");
	};
	const auto consume = [&](const char * consumer)
	{
		contextState.executing = true;
		contextState.commands = 0;
		contextState.routeInteractions.reset();
		auto record = contextState.record("consumed", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["consumer"].String() = consumer;
		contextState.records.push_back(std::move(record));
	};
	const auto changePanel = [&](const std::string & panel, const std::string & selected)
	{
		contextState.panel = panel;
		contextState.selection = selected;
		auto record = contextState.record("context", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["context"] = contextState.contextJson();
		record["revision"].Integer() = ++contextState.revision;
		contextState.records.push_back(std::move(record));
	};
	{
		const auto gameLock = readGameState();
		captureContextual();
		std::lock_guard lock(stateMutex);
		if(contextState.decision != decision || terminal || stopping) return false;
		if(!failure.empty()) throw std::runtime_error(failure);
		const auto * town = activeTown();
		validate(*town);
		if(!buying)
		{
			consume("adapter");
			if(opening)
			{
				contextState.economy.selectedRecruitIndex = 0;
				changePanel("recruitment", selection);
			}
			else if(kind == "back") changePanel("overview", "");
			else
			{
				const auto & offers = contextState.lastObservation["window"]["town_panel"]["recruitment"]["offers"].Vector();
				const auto found = std::find_if(offers.begin(), offers.end(), [&](const auto & offer) { return offer["ref"].String() == selection; });
				contextState.economy.selectedRecruitIndex = static_cast<size_t>(std::distance(offers.begin(), found));
				changePanel("recruitment", contextState.selection);
			}
		}
	}
	if(!buying)
	{
		contextualResult(recruitmentResult(opening ? "opened" : kind == "back" ? "closed" : "selected", true), decision);
		return true;
	}

	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto * town = activeTown();
		const auto * destination = town->getUpperArmy();
		if(town->id != townID || destination->id != destinationID || destination->getOwner() != playerID)
			throw std::runtime_error("Revalidated recruitment destination changed");
		callback->recruitCreatures(town, destination, creatureID, static_cast<ui32>(count), level);
		// Native CRecruitmentWindow::buy closes a level-specific popup after
		// onRecruit, before the acknowledgement and incoming state mutations.
		changePanel("overview", "");
	}, true, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto * town = activeTown();
		validate(*town);
		level = contextState.economy.resolveDwelling(*town, contextState.selection);
		const auto & pool = town->creatures[level];
		const auto selectedIndex = contextState.economy.selectedRecruitIndex;
		if(selectedIndex >= pool.second.size()) throw std::runtime_error("Recruitment card changed before issue");
		creatureID = pool.second[pool.second.size() - 1 - selectedIndex];
		const auto * creature = creatureID.toCreature();
		const auto * destination = town->getUpperArmy();
		if(!creature || creature->warMachine.hasValue() || !destination || destination->getOwner() != playerID
			|| contextState.economy.recruitHandle(*town, level, creatureID) != selection
			|| contextState.economy.destinationHandle(*town) != action["destination"].String()
			|| count > pool.first || count > creature->maxAmount(callback->getResourceAmount()))
			throw std::runtime_error("Recruitment changed before issue");
		// The server clamps overstock quantities. This guard shares the lock
		// with the actual send, so the requested count can never be substituted.
		slot = destination->getSlotFor(creatureID);
		if(!slot.validSlot())
		{
			consume("adapter");
			std::pair<SlotID, SlotID> toMerge;
			if(callback->getSettings().getBoolean(EGameSettings::DWELLINGS_ACCUMULATE_WHEN_OWNED)
				&& destination->mergeableStacks(toMerge))
				throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: recruitment auto-merge");
			noCapacity = true;
			auto event = contextState.record("event", callback->getDate(Date::DAY));
			event["event"] = explorer_rl::recruitmentCapacityNotice();
			++contextState.interactions;
			contextState.records.push_back(std::move(event));
			return false;
		}
		townID = town->id;
		destinationID = destination->id;
		beforePools = town->creatures;
		beforeArmy = recruitmentArmy(*destination);
		if(const auto existing = beforeArmy.find(slot.getNum()); existing != beforeArmy.end()
			&& existing->second.second + count > std::numeric_limits<int32_t>::max())
			throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: recruitment stack overflow");
		const auto & cost = creature->getFullRecruitCost();
		for(int resource = 0; resource < 7; ++resource)
		{
			resources[resource] = callback->getResourceAmount(GameResID(resource));
			debit[resource] = static_cast<int64_t>(cost[resource]) * count;
			if(debit[resource] < 0 || debit[resource] > resources[resource])
				throw std::runtime_error("Recruitment cost is not affordable");
		}
		consume("engine");
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(noCapacity)
	{
		contextualResult(recruitmentResult("no_capacity", false), decision);
		return true;
	}
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		const auto * town = activeTown();
		const auto * destination = town->getUpperArmy();
		if(!destination || town->id != townID || destination->id != destinationID || destination->getOwner() != playerID)
			throw std::runtime_error("Recruitment acknowledgement changed its destination");
		auto expectedPools = beforePools;
		auto expectedArmy = beforeArmy;
		if(accepted)
		{
			expectedPools[level].first -= static_cast<ui32>(count);
			auto & stack = expectedArmy[slot.getNum()];
			stack.first = creatureID.getNum();
			stack.second += count;
		}
		if(town->creatures != expectedPools || recruitmentArmy(*destination) != expectedArmy)
			throw std::runtime_error("Recruitment acknowledgement does not match exact stock and army effects");
		for(int resource = 0; resource < 7; ++resource)
			if(resources[resource] - callback->getResourceAmount(GameResID(resource)) != (accepted ? debit[resource] : 0))
				throw std::runtime_error("Recruitment acknowledgement does not match exact resource effects");
	}
	contextualResult(recruitmentResult(accepted ? "recruited" : "rejected", accepted), decision);
	return true;
}
