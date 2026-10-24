#include "StdInc.h"
#include "ExplorerRL.h"
#include "TransferRules.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"

#include <limits>

namespace
{
explorer_rl::TransferArmy transferArmy(const CArmedInstance & army)
{
	explorer_rl::TransferArmy result;
	for(const auto & [slot, stack] : army.Slots())
	{
		if(!slot.validSlot() || !stack || !stack->getCreature() || stack->getCount() <= 0)
			throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: transfer army slots");
		result.at(slot.getNum()) = {stack->getCreatureID().getNum(), stack->getCount()};
	}
	return result;
}

JsonNode transferResult(const char * status, bool accepted)
{
	JsonNode result;
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	for(const char * field : {"position", "movement", "destination_reached"}) result[field] = JsonNode();
	return result;
}
}

JsonNode ExplorerRL::townArmyExchange(const CGTownInstance & town)
{
	// These are precisely the two rows displayed by CCastleInterface's
	// CGarrisonInt. Never choose a nearby hero or inspect its private window.
	const auto * source = town.getUpperArmy();
	const auto * destination = town.getVisitingHero();
	if(town.getOwner() != playerID || !source || source->getOwner() != playerID
		|| (destination && destination->getOwner() != playerID))
		throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: non-owned town army row");
	const auto row = [&](const CArmedInstance & object, const char * role)
	{
		JsonNode result;
		const auto key = std::pair(town.id.getNum(), object.id.getNum());
		auto [armyRef, inserted] = contextState.armyHandles.try_emplace(key);
		if(inserted) armyRef->second = "army-" + std::to_string(contextState.armyHandles.size() - 1);
		result["ref"].String() = armyRef->second;
		result["role"].String() = role;
		result["subject"].String() = contextState.handle(object.id.getNum());
		result["requires_nonempty"].Bool() = object.needsLastStack();
		const auto stacks = transferArmy(object);
		for(int index = 0; index < 7; ++index)
		{
			const auto slotKey = std::tuple(town.id.getNum(), object.id.getNum(), index);
			auto [slotRef, newSlot] = contextState.armySlotHandles.try_emplace(slotKey);
			if(newSlot) slotRef->second = "army-slot-" + std::to_string(contextState.armySlotHandles.size() - 1);
			JsonNode slot;
			slot["ref"].String() = slotRef->second;
			slot["slot"].Integer() = index;
			slot["creature"] = stacks[index].count ? JsonNode(stacks[index].creature) : JsonNode();
			slot["count"].Integer() = stacks[index].count;
			result["slots"].Vector().push_back(std::move(slot));
		}
		return result;
	};
	JsonNode result;
	result["source"] = row(*source, source == &town ? "town_garrison" : "garrison_hero");
	result["destination"] = destination ? row(*destination, "visiting_hero") : JsonNode();
	return result;
}

bool ExplorerRL::executeTransferAction(const JsonNode & action, int64_t decision)
{
	if(!armyTransfer || !action["subject"].isString() || !action["selection"].isString()
		|| !action["destination"].isString() || !action["x"].isNull() || !action["y"].isNull()
		|| !action["panel"].isNull() || action["count"].getType() != JsonNode::JsonType::DATA_INTEGER
		|| action["count"].Integer() <= 0 || action["count"].Integer() > std::numeric_limits<int32_t>::max())
		throw std::runtime_error("Transfer requires explicit town, source slot, destination slot and positive count");
	const auto subject = action["subject"].String();
	const int64_t count = action["count"].Integer();
	ObjectInstanceID townID(-1), sourceID(-1), destinationID(-1);
	int sourceSlot = -1, destinationSlot = -1;
	int64_t revision = -1;
	std::array<int64_t, 7> resources{};
	explorer_rl::TransferArmy beforeSource, beforeDestination;
	explorer_rl::TransferEffect expected;
	const auto activeTown = [&]() -> const CGTownInstance *
	{
		if(contextState.context != "town" || contextState.subject != subject || contextState.panel != "overview")
			throw std::runtime_error("Transfer requires its explicit town overview");
		const auto handle = contextState.subjects.find(subject);
		if(handle == contextState.subjects.end()) throw std::runtime_error("Unknown transfer town");
		for(const auto * town : callback->getTownsInfo())
			if(town->id.getNum() == handle->second && town->getOwner() == playerID) return town;
		throw std::runtime_error("Transfer town is no longer owned");
	};
	const auto ownedRows = [&](const CGTownInstance & town)
	{
		const auto * source = town.getUpperArmy();
		const auto * destination = town.getVisitingHero();
		if(!source || !destination || source == destination || source->getOwner() != playerID
			|| destination->getOwner() != playerID)
			throw std::runtime_error("Transfer requires the actual owned upper army and visiting hero");
		return std::pair(source, destination);
	};
	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto * town = activeTown();
		const auto [source, destination] = ownedRows(*town);
		if(town->id != townID || source->id != sourceID || destination->id != destinationID)
			throw std::runtime_error("Transfer army identity changed before native issue");
		// Native ArrangeStacks.what=3 means final destination total, not a
		// quantity to move. The external positive count is never substituted.
		callback->splitStack(source, destination, SlotID(sourceSlot), SlotID(destinationSlot), expected.destinationTotal);
	}, true, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto * town = activeTown();
		const auto [source, destination] = ownedRows(*town);
		const auto & armies = contextState.lastObservation["window"]["town_panel"]["armies"];
		const auto resolve = [](const JsonNode & row, const JsonNode & selected)
		{
			for(const auto & slot : row["slots"].Vector())
				if(slot["ref"] == selected) return static_cast<int>(slot["slot"].Integer());
			throw std::runtime_error("Transfer slot was not displayed in the selected row");
		};
		sourceSlot = resolve(armies["source"], action["selection"]);
		destinationSlot = resolve(armies["destination"], action["destination"]);
		beforeSource = transferArmy(*source);
		beforeDestination = transferArmy(*destination);
		try
		{
			expected = explorer_rl::transferEffect(beforeSource, beforeDestination, sourceSlot, destinationSlot,
				count, source->needsLastStack());
		}
		catch(const std::overflow_error &)
		{
			throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: transfer stack overflow");
		}
		townID = town->id;
		sourceID = source->id;
		destinationID = destination->id;
		revision = contextState.revision;
		for(int resource = 0; resource < 7; ++resource) resources[resource] = callback->getResourceAmount(GameResID(resource));
		contextState.executing = true;
		contextState.commands = 0;
		contextState.routeInteractions.reset();
		auto consumed = contextState.record("consumed", callback->getDate(Date::DAY));
		consumed["decision_id"].Integer() = decision;
		consumed["consumer"].String() = "engine";
		contextState.records.push_back(std::move(consumed));
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		const auto * town = activeTown();
		const auto [source, destination] = ownedRows(*town);
		if(town->id != townID || source->id != sourceID || destination->id != destinationID
			|| contextState.revision != revision
			|| transferArmy(*source) != (accepted ? expected.source : beforeSource)
			|| transferArmy(*destination) != (accepted ? expected.destination : beforeDestination))
			throw std::runtime_error("Transfer acknowledgement does not match exact selected army effects");
		for(int resource = 0; resource < 7; ++resource)
			if(callback->getResourceAmount(GameResID(resource)) != resources[resource])
				throw std::runtime_error("Transfer unexpectedly changed player resources");
	}
	contextualResult(transferResult(accepted ? "transferred" : "rejected", accepted), decision);
	return true;
}
