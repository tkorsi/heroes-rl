#include "StdInc.h"
#include "ExplorerRL.h"
#include "ArmyOperationRules.h"
#include "GarrisonRules.h"

#include "../../lib/IGameSettings.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/gameState/UpgradeInfo.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"

#include <algorithm>
#include <limits>

namespace
{
using explorer_rl::ArmyOperation;
using explorer_rl::TransferArmy;
using State = explorer_rl::ContextualState;
using Resources = std::array<int64_t, 7>;
using ExchangeCheck = std::function<bool(int, int)>;
struct Row { const CArmedInstance * army; const char * role; bool removable = true; };
using Rows = std::vector<Row>;
using Armies = std::map<int, TransferArmy>;

int subjectID(const State & state, const std::string & subject)
{
	const auto found = state.subjects.find(subject);
	if(found == state.subjects.end()) throw std::runtime_error("Unknown owned army subject");
	return found->second;
}

const CGHeroInstance * ownedHero(const CCallback & callback, int id)
{
	for(const auto * hero : callback.getHeroesInfo())
		if(hero->id.getNum() == id && hero->getOwner() == callback.getPlayerID()) return hero;
	throw std::runtime_error("Army hero is no longer owned");
}

const CGTownInstance * ownedTown(const CCallback & callback, int id)
{
	for(const auto * town : callback.getTownsInfo())
		if(town->id.getNum() == id && town->getOwner() == callback.getPlayerID()) return town;
	throw std::runtime_error("Army town is no longer owned");
}

Rows displayedRows(const CCallback & callback, const State & state, const JsonNode & context,
	const ExchangeCheck & exchange)
{
	Rows rows;
	const auto & kind = context["kind"].String();
	if(kind == "hero")
		rows.push_back({ownedHero(callback, subjectID(state, context["subject"].String())), "hero"});
	else if(kind == "town" && context["panel"].String() == "overview")
	{
		const auto * town = ownedTown(callback, subjectID(state, context["subject"].String()));
		rows.push_back({town->getUpperArmy(), town->getGarrisonHero() ? "garrison_hero" : "town_garrison"});
		if(const auto * visitor = town->getVisitingHero()) rows.push_back({visitor, "visiting_hero"});
	}
	else if(kind == "exchange")
	{
		const int first = subjectID(state, context["subject"].String());
		const int second = subjectID(state, context["selection"].String());
		if(first == second || !exchange(first, second))
			throw std::runtime_error("Army exchange requires its active native owned-hero query");
		rows.push_back({ownedHero(callback, first), "hero"});
		if(state.ownHeroExchange && state.ownHeroExchange->joiningNeutral && state.ownHeroExchange->second == second)
		{
			const auto * joining = dynamic_cast<const CArmedInstance *>(callback.getObj(ObjectInstanceID(second), false));
			if(!joining || joining->ID != Obj::MONSTER || joining->getOwner().isValidPlayer())
				throw std::runtime_error("Received joining army is unavailable");
			rows.push_back({joining, "joining_neutral"});
		}
		else if(state.ownHeroExchange && state.ownHeroExchange->objectGarrison && state.ownHeroExchange->second == second)
		{
			const auto & received = *state.ownHeroExchange;
			const auto * garrison = dynamic_cast<const CArmedInstance *>(callback.getObj(ObjectInstanceID(second), false));
			const bool self = garrison && garrison->getOwner() == callback.getPlayerID();
			const bool allied = garrison && garrison->getOwner().isValidPlayer()
				&& callback.getPlayerID() && callback.getPlayerRelations(*callback.getPlayerID(), garrison->getOwner()) == PlayerRelations::ALLIES;
			if(!garrison || (garrison->ID != Obj::MINE && garrison->ID != Obj::GARRISON && garrison->ID != Obj::GARRISON2)
				|| (received.allied ? !allied : !self))
				throw std::runtime_error("Received object garrison is unavailable or changed ownership");
			rows.push_back({garrison, explorer_rl::objectGarrisonRole(received.allied, received.removableUnits), received.removableUnits});
		}
		else rows.push_back({ownedHero(callback, second), "hero"});
	}
	else throw std::runtime_error("Army operations require an explicitly opened army window");
	for(const auto & row : rows)
		if(!row.army || (row.army->getOwner() != callback.getPlayerID() && std::string(row.role) != "joining_neutral"
			&& std::string(row.role) != "allied_garrison" && std::string(row.role) != "allied_locked_garrison"))
			throw std::runtime_error("Displayed army row is not owned");
	return rows;
}

explorer_rl::ArmyAccess access(const CCallback & callback, const Rows & rows, const CArmedInstance * army)
{
	for(const auto & row : rows)
		if(row.army == army)
			return {army->getOwner() == callback.getPlayerID(), army->getOwner() == PlayerColor::UNFLAGGABLE, row.removable};
	throw std::runtime_error("Army permissions require its received displayed row");
}

TransferArmy snapshot(const CArmedInstance & army)
{
	TransferArmy result{};
	for(const auto & [slot, stack] : army.Slots())
	{
		if(!slot.validSlot() || !stack || !stack->getCreature())
			throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: army slots");
		result.at(slot.getNum()) = {stack->getCreatureID().getNum(), stack->getCount()};
	}
	explorer_rl::validateArmy(result);
	return result;
}

Armies snapshot(const Rows & rows)
{
	Armies result;
	for(const auto & row : rows)
		if(!result.emplace(row.army->id.getNum(), snapshot(*row.army)).second)
			throw std::runtime_error("Duplicate displayed army");
	return result;
}

Resources resources(const CCallback & callback)
{
	Resources result{};
	for(int i = 0; i < 7; ++i) result[i] = callback.getResourceAmount(GameResID(i));
	return result;
}

JsonNode resultNode(const char * status, bool accepted)
{
	JsonNode result;
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	for(const char * key : {"position", "movement", "destination_reached"}) result[key] = JsonNode();
	return result;
}

void consume(State & state, int day, int64_t decision, const char * consumer)
{
	state.executing = true;
	state.commands = 0;
	state.routeInteractions.reset();
	auto record = state.record("consumed", day);
	record["decision_id"].Integer() = decision;
	record["consumer"].String() = consumer;
	state.records.push_back(std::move(record));
}

void changeContext(State & state, const JsonNode & context, int day, int64_t decision)
{
	state.context = context["kind"].String();
	state.subject = context["subject"].isString() ? context["subject"].String() : "";
	state.panel = context["panel"].isString() ? context["panel"].String() : "";
	state.selection = context["selection"].isString() ? context["selection"].String() : "";
	auto record = state.record("context", day);
	record["decision_id"].Integer() = decision;
	record["context"] = state.contextJson();
	record["revision"].Integer() = ++state.revision;
	state.records.push_back(std::move(record));
}

JsonNode slotView(State & state, int anchor, const CArmedInstance & army, int index)
{
	const auto key = std::tuple(anchor, army.id.getNum(), index);
	auto [ref, inserted] = state.armySlotHandles.try_emplace(key);
	if(inserted) ref->second = "army-slot-" + std::to_string(state.armySlotHandles.size() - 1);
	JsonNode result;
	result["ref"].String() = ref->second;
	result["slot"].Integer() = index;
	const auto * stack = army.getStackPtr(SlotID(index));
	if(stack && (!stack->getCreature() || stack->getCount() <= 0))
		throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: stack details");
	result["creature"] = stack ? JsonNode(stack->getCreatureID().getNum()) : JsonNode();
	result["count"].Integer() = stack ? stack->getCount() : 0;
	return result;
}

const CArmedInstance * rowArmy(const State & state, const Rows & rows, const std::string & subject)
{
	const int id = subjectID(state, subject);
	for(const auto & row : rows) if(row.army->id.getNum() == id) return row.army;
	throw std::runtime_error("Selected army is not in the current displayed rows");
}

std::pair<const CArmedInstance *, int> selectedSlot(const State & state, const Rows & rows,
	const JsonNode & wireRows, const std::string & ref)
{
	for(const auto & row : wireRows.Vector())
		for(const auto & slot : row["slots"].Vector())
			if(slot["ref"].String() == ref)
				return {rowArmy(state, rows, row["subject"].String()), static_cast<int>(slot["slot"].Integer())};
	throw std::runtime_error("Selected slot was not displayed in the current army window");
}

std::pair<const CArmedInstance *, int> openedStack(const CCallback & callback, const State & state,
	const ExchangeCheck & exchange)
{
	if(state.context != "stack" || (state.panel != "overview" && state.panel != "upgrade"))
		throw std::runtime_error("Stack details require the explicitly opened stack");
	const auto rows = displayedRows(callback, state, state.stackParentContext, exchange);
	const auto * army = rowArmy(state, rows, state.subject);
	const int anchor = subjectID(state, state.stackParentContext["subject"].String());
	for(int slot = 0; slot < 7; ++slot)
	{
		const auto found = state.armySlotHandles.find(std::tuple(anchor, army->id.getNum(), slot));
		if(found != state.armySlotHandles.end() && found->second == state.selection)
		{
			if(!army->hasStackAtSlot(SlotID(slot))) throw std::runtime_error("Opened stack is no longer present");
			return {army, slot};
		}
	}
	throw std::runtime_error("Opened stack no longer matches its displayed parent slot");
}

UpgradeInfo upgrades(const CCallback & callback, const CArmedInstance & army, int slot)
{
	UpgradeInfo result(army.getStackPtr(SlotID(slot))->getCreatureID());
	if(army.getOwner() == callback.getPlayerID()) callback.fillUpgradeInfo(&army, SlotID(slot), result);
	return result;
}

Resources upgradeCost(const UpgradeInfo & info, CreatureID creature, int64_t count)
{
	// Validate membership before getUpgradeCostsFor, which asserts on bad IDs.
	const auto & offered = info.getAvailableUpgrades();
	const size_t displayed = std::min<size_t>(offered.size(), 3); // native CStackWindow button capacity
	if(std::find(offered.begin(), offered.begin() + displayed, creature) == offered.begin() + displayed)
		throw std::runtime_error("Upgrade target is not a displayed native button");
	Resources perCreature{};
	const auto & costs = info.getUpgradeCostsFor(creature);
	for(int i = 0; i < 7; ++i) perCreature[i] = costs[i];
	return explorer_rl::upgradeCost(perCreature, count);
}

JsonNode garrisonNotice(const char * textID, int heroCount = -1)
{
	JsonNode event;
	event["kind"].String() = "notice";
	auto & notice = event["notice"];
	notice["display_mode"].String() = "modal";
	notice["components"].Vector();
	JsonNode text;
	text["op"].String() = "append_text";
	text["text_id"].String() = textID;
	text["number"] = JsonNode();
	text["owner"] = JsonNode();
	notice["text"].Vector().push_back(std::move(text));
	if(heroCount >= 0)
	{
		JsonNode count;
		count["op"].String() = "replace_number";
		count["text_id"] = JsonNode();
		count["number"].Integer() = heroCount;
		count["owner"] = JsonNode();
		notice["text"].Vector().push_back(std::move(count));
	}
	return event;
}
}

JsonNode ExplorerRL::contextualArmyViews()
{
	if(!armyManagement) throw std::runtime_error("Army views require the v6 profile");
	JsonNode result;
	result.Vector();
	if(contextState.context != "hero" && contextState.context != "exchange"
		&& !(contextState.context == "town" && contextState.panel == "overview"))
		return result; // an overlay never refreshes the hidden parent's rows
	const auto rows = displayedRows(*callback, contextState, contextState.contextJson(),
		[this](int first, int second) { return activeOwnHeroExchange(first, second); });
	const int anchor = subjectID(contextState, contextState.subject);
	for(const auto & row : rows)
	{
		snapshot(*row.army); // validate native supported quantities before exposing them
		const auto key = std::pair(anchor, row.army->id.getNum());
		auto [ref, inserted] = contextState.armyHandles.try_emplace(key);
		if(inserted) ref->second = "army-" + std::to_string(contextState.armyHandles.size() - 1);
		JsonNode view;
		view["ref"].String() = ref->second;
		view["role"].String() = row.role;
		view["subject"].String() = contextState.handle(row.army->id.getNum());
		view["requires_nonempty"].Bool() = row.army->needsLastStack();
		for(int slot = 0; slot < 7; ++slot) view["slots"].Vector().push_back(slotView(contextState, anchor, *row.army, slot));
		result.Vector().push_back(std::move(view));
	}
	return result;
}

bool ExplorerRL::executeArmyAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool splitting = kind == "split_stack";
	if(!armyManagement || (!splitting && kind != "merge_stacks" && kind != "swap_stacks")
		|| !action["subject"].isString() || !action["selection"].isString() || !action["destination"].isString()
		|| !action["x"].isNull() || !action["y"].isNull() || !action["panel"].isNull()
		|| (splitting ? action["count"].getType() != JsonNode::JsonType::DATA_INTEGER : !action["count"].isNull()))
		throw std::runtime_error("Incorrect explicit army action parameters");
	const auto operation = splitting ? ArmyOperation::SPLIT : kind == "merge_stacks" ? ArmyOperation::MERGE : ArmyOperation::SWAP;
	const auto subject = action["subject"].String();
	const int64_t count = splitting ? action["count"].Integer() : 0;
	if(splitting && (count < 1 || count > std::numeric_limits<int32_t>::max()))
		throw std::runtime_error("Split requires an explicit positive native count");
	const auto rows = [&]()
	{
		if(contextState.subject != subject) throw std::runtime_error("Army action does not match the active window");
		return displayedRows(*callback, contextState, contextState.contextJson(),
			[this](int first, int second) { return activeOwnHeroExchange(first, second); });
	};
	bool waitForAdventure;
	{
		std::lock_guard lock(stateMutex);
		waitForAdventure = contextState.context != "exchange";
	}
	int sourceID = -1, destinationID = -1, sourceSlot = -1, destinationSlot = -1;
	int64_t revision = -1;
	Armies before, expected;
	Resources beforeResources{};
	explorer_rl::ArmyOperationEffect effect;
	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto current = rows();
		const CArmedInstance * source = nullptr, * destination = nullptr;
		for(const auto & row : current)
		{
			if(row.army->id.getNum() == sourceID) source = row.army;
			if(row.army->id.getNum() == destinationID) destination = row.army;
		}
		if(!source || !destination || snapshot(current) != before)
			throw std::runtime_error("Selected armies changed before native issue");
		if(splitting) callback->splitStack(source, destination, SlotID(sourceSlot), SlotID(destinationSlot), effect.destinationTotal);
		else if(operation == ArmyOperation::MERGE) callback->mergeStacks(source, destination, SlotID(sourceSlot), SlotID(destinationSlot));
		else callback->swapCreatures(source, destination, SlotID(sourceSlot), SlotID(destinationSlot));
	}, waitForAdventure, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto current = rows();
		const auto & wire = contextState.lastObservation["window"]["armies"];
		const auto [source, from] = selectedSlot(contextState, current, wire, action["selection"].String());
		const auto [destination, to] = selectedSlot(contextState, current, wire, action["destination"].String());
		sourceID = source->id.getNum(); destinationID = destination->id.getNum();
		sourceSlot = from; destinationSlot = to;
		before = snapshot(current);
		explorer_rl::validateArmyOperationAccess(operation, sourceID == destinationID,
			before.at(sourceID).at(from).count, before.at(destinationID).at(to).count,
			access(*callback, current, source), access(*callback, current, destination));
		effect = explorer_rl::armyOperationEffect(operation, before.at(sourceID), before.at(destinationID),
			from, to, count, sourceID == destinationID, source->needsLastStack(), destination->needsLastStack());
		expected = before;
		expected[sourceID] = effect.source;
		expected[destinationID] = effect.destination;
		beforeResources = resources(*callback);
		revision = contextState.revision;
		consume(contextState, callback->getDate(Date::DAY), decision, "engine");
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		if(contextState.revision != revision || snapshot(rows()) != (accepted ? expected : before)
			|| resources(*callback) != beforeResources)
			throw std::runtime_error("Army acknowledgement does not match exact selected slots and resources");
	}
	contextualResult(resultNode(accepted ? splitting ? "split" : operation == ArmyOperation::MERGE ? "merged" : "swapped" : "rejected", accepted), decision);
	return true;
}

JsonNode ExplorerRL::stackPanel()
{
	if(!armyManagement) throw std::runtime_error("Stack panel requires the v6 profile");
	const auto [army, slot] = openedStack(*callback, contextState,
		[this](int first, int second) { return activeOwnHeroExchange(first, second); });
	const int anchor = subjectID(contextState, contextState.stackParentContext["subject"].String());
	JsonNode result;
	result["parent_context"] = contextState.stackParentContext;
	result["source"] = slotView(contextState, anchor, *army, slot);
	result["offers"].Vector();
	for(const char * key : {"selected_upgrade_ref", "total_cost", "affordable"}) result[key] = JsonNode();
	const auto info = upgrades(*callback, *army, slot);
	const auto & offered = info.getAvailableUpgrades();
	const size_t displayed = std::min<size_t>(offered.size(), 3);
	for(size_t i = 0; i < displayed; ++i)
	{
		auto [ref, inserted] = contextState.upgradeHandles.try_emplace(offered[i].getNum());
		if(inserted) ref->second = "upgrade-" + std::to_string(contextState.upgradeHandles.size() - 1);
		JsonNode choice;
		choice["ref"].String() = ref->second;
		choice["creature"].Integer() = offered[i].getNum();
		result["offers"].Vector().push_back(std::move(choice));
		if(contextState.panel == "upgrade" && contextState.selectedUpgradeRef == ref->second)
		{
			const auto cost = upgradeCost(info, offered[i], army->getStackCount(SlotID(slot)));
			const auto current = resources(*callback);
			bool affordable = true;
			for(size_t resource = 0; resource < cost.size(); ++resource)
			{
				result["total_cost"].Vector().emplace_back(cost[resource]);
				affordable = affordable && current[resource] >= cost[resource];
			}
			result["selected_upgrade_ref"].String() = ref->second;
			result["affordable"].Bool() = affordable;
		}
	}
	if(contextState.panel == "upgrade" && result["selected_upgrade_ref"].isNull())
		throw std::runtime_error("Selected upgrade is no longer displayed");
	return result;
}

bool ExplorerRL::executeStackAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool opening = kind == "open_stack", selecting = kind == "select_upgrade", buying = kind == "upgrade_stack";
	if(!armyManagement || (!opening && !selecting && !buying && kind != "back")
		|| !action["subject"].isString() || !action["x"].isNull() || !action["y"].isNull()
		|| !action["panel"].isNull() || !action["count"].isNull() || !action["destination"].isNull()
		|| (kind == "back" ? !action["selection"].isNull() : !action["selection"].isString()))
		throw std::runtime_error("Incorrect explicit stack action parameters");
	const auto subject = action["subject"].String();
	const auto selection = action["selection"].isString() ? action["selection"].String() : "";
	const auto exchange = [this](int first, int second) { return activeOwnHeroExchange(first, second); };
	const auto validate = [&]()
	{
		if(contextState.subject != subject) throw std::runtime_error("Stack action does not match its active subject");
		if(opening) return;
		openedStack(*callback, contextState, exchange);
		if(kind == "back") return;
		const auto & panel = contextState.lastObservation["window"]["stack_panel"];
		if(selecting && contextState.panel != "overview")
			throw std::runtime_error("Upgrade selection requires the opened stack overview");
		const auto & offers = panel["offers"].Vector();
		if(std::none_of(offers.begin(), offers.end(), [&](const auto & item) { return item["ref"].String() == selection; }))
			throw std::runtime_error("Upgrade choice was not displayed");
		if(buying && (contextState.panel != "upgrade" || panel["selected_upgrade_ref"].String() != selection
			|| !panel["affordable"].Bool()))
			throw std::runtime_error("Upgrade acceptance requires the selected affordable confirmation");
	};
	JsonNode parent;
	{
		const auto gameLock = readGameState();
		captureContextual();
		std::lock_guard lock(stateMutex);
		if(contextState.decision != decision || terminal || stopping) return false;
		if(!failure.empty()) throw std::runtime_error(failure);
		validate();
		if(!buying)
		{
			JsonNode next;
			if(opening)
			{
				parent = contextState.contextJson();
				const auto rows = displayedRows(*callback, contextState, parent, exchange);
				const auto [army, slot] = selectedSlot(contextState, rows, contextState.lastObservation["window"]["armies"], selection);
				if(!army->hasStackAtSlot(SlotID(slot))) throw std::runtime_error("Cannot open an empty stack slot");
				contextState.stackParentContext = parent;
				contextState.selectedUpgradeRef.clear();
				next["kind"].String() = "stack";
				next["subject"].String() = contextState.handle(army->id.getNum());
				next["selection"].String() = selection;
				next["panel"].String() = "overview";
			}
			else if(selecting)
			{
				contextState.selectedUpgradeRef = selection;
				next = contextState.contextJson();
				next["panel"].String() = "upgrade";
			}
			else if(contextState.panel == "upgrade")
			{
				contextState.selectedUpgradeRef.clear();
				next = contextState.contextJson();
				next["panel"].String() = "overview";
			}
			else
			{
				next = contextState.stackParentContext;
				contextState.stackParentContext = JsonNode();
				contextState.selectedUpgradeRef.clear();
			}
			consume(contextState, callback->getDate(Date::DAY), decision, "adapter");
			changeContext(contextState, next, callback->getDate(Date::DAY), decision);
		}
		else parent = contextState.stackParentContext;
	}
	if(!buying)
	{
		contextualResult(resultNode(opening ? "opened" : selecting ? "selected" : "closed", true), decision);
		return true;
	}
	int armyID = -1, slot = -1;
	CreatureID upgrade(-1);
	Armies before, expected;
	Resources beforeResources{}, debit{};
	int64_t revision = -1;
	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto [army, currentSlot] = openedStack(*callback, contextState, exchange);
		if(army->id.getNum() != armyID || currentSlot != slot) throw std::runtime_error("Upgrade stack changed before issue");
		callback->upgradeCreature(army, SlotID(slot), upgrade);
		// Native CStackWindow closes immediately after its explicit Yes callback.
		changeContext(contextState, parent, callback->getDate(Date::DAY), decision);
		contextState.stackParentContext = JsonNode();
		contextState.selectedUpgradeRef.clear();
	}, parent["kind"].String() != "exchange", decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		validate();
		const auto [army, currentSlot] = openedStack(*callback, contextState, exchange);
		armyID = army->id.getNum(); slot = currentSlot;
		const auto & panel = contextState.lastObservation["window"]["stack_panel"];
		for(const auto & choice : panel["offers"].Vector())
			if(choice["ref"].String() == selection) upgrade = CreatureID(choice["creature"].Integer());
		const auto info = upgrades(*callback, *army, slot);
		debit = upgradeCost(info, upgrade, army->getStackCount(SlotID(slot)));
		beforeResources = resources(*callback);
		for(int i = 0; i < 7; ++i)
			if(debit[i] != panel["total_cost"].Vector().at(i).Integer() || debit[i] > beforeResources[i])
				throw std::runtime_error("Upgrade displayed confirmation cost changed before issue");
		// Receipt-only snapshots of the already opened parent rows. They do
		// not refresh a closed parent's private contents in the learner view.
		before = snapshot(displayedRows(*callback, contextState, parent, exchange));
		expected = before;
		expected.at(armyID).at(slot).creature = upgrade.getNum();
		revision = contextState.revision;
		consume(contextState, callback->getDate(Date::DAY), decision, "engine");
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		if(contextState.revision != revision + 1 || contextState.contextJson() != parent
			|| snapshot(displayedRows(*callback, contextState, parent, exchange)) != (accepted ? expected : before))
			throw std::runtime_error("Upgrade acknowledgement does not match exact stack type and count");
		const auto after = resources(*callback);
		for(int i = 0; i < 7; ++i)
			if(beforeResources[i] - after[i] != (accepted ? debit[i] : 0))
				throw std::runtime_error("Upgrade acknowledgement does not match exact resource debit");
	}
	contextualResult(resultNode(accepted ? "upgraded" : "rejected", accepted), decision);
	return true;
}

bool ExplorerRL::executeGarrisonAction(const JsonNode & action, int64_t decision)
{
	if(!armyManagement || action["kind"].String() != "swap_garrison" || !action["subject"].isString())
		throw std::runtime_error("Incorrect garrison action");
	for(const char * key : {"x", "y", "selection", "panel", "count", "destination"})
		if(!action[key].isNull()) throw std::runtime_error("Unused garrison action parameter");
	const auto subject = action["subject"].String();
	const auto activeTown = [&]()
	{
		if(contextState.context != "town" || contextState.panel != "overview" || contextState.subject != subject)
			throw std::runtime_error("Garrison action requires its opened town overview");
		return ownedTown(*callback, subjectID(contextState, subject));
	};
	int townID = -1, visitorID = -1, garrisonID = -1;
	Armies before, expected;
	Resources beforeResources{};
	int64_t revision = -1;
	bool issued = false, blocked = false;
	const bool accepted = executeRequest([&]
	{
		const auto * town = activeTown();
		if(town->id.getNum() != townID) throw std::runtime_error("Garrison town changed before issue");
		callback->swapGarrisonHero(town);
	}, true, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto * town = activeTown();
		const auto * visitor = town->getVisitingHero();
		const auto * garrison = town->getGarrisonHero();
		if((!visitor && !garrison) || (visitor && visitor->getOwner() != playerID)
			|| (garrison && garrison->getOwner() != playerID))
			throw std::runtime_error("Garrison action requires displayed owned hero slots");
		JsonNode notice;
		// Exactly HeroSlots::swapArmies' on-click checks, not a hidden mask.
		if(garrison && !visitor && callback->howManyHeroes(false)
			>= callback->getSettings().getInteger(EGameSettings::HEROES_PER_PLAYER_ON_MAP_CAP))
			notice = garrisonNotice("core.genrltxt.18", callback->howManyHeroes(false));
		else if(garrison && garrison->stacksCount() == 0) notice = garrisonNotice("core.genrltxt.19");
		else if(!garrison && visitor && !visitor->canBeMergedWith(*town)) notice = garrisonNotice("core.genrltxt.275");
		if(!notice.isNull())
		{
			consume(contextState, callback->getDate(Date::DAY), decision, "adapter");
			auto record = contextState.record("event", callback->getDate(Date::DAY));
			record["event"] = std::move(notice);
			contextState.records.push_back(std::move(record));
			++contextState.interactions;
			blocked = true;
			return false;
		}
		townID = town->id.getNum();
		visitorID = visitor ? visitor->id.getNum() : -1;
		garrisonID = garrison ? garrison->id.getNum() : -1;
		before.emplace(townID, snapshot(*town));
		if(visitor) before.emplace(visitorID, snapshot(*visitor));
		if(garrison) before.emplace(garrisonID, snapshot(*garrison));
		expected = before;
		if(visitor && !garrison)
		{
			const auto effect = explorer_rl::garrisonArmyEffect(before.at(townID), before.at(visitorID));
			expected[townID] = effect.source;
			expected[visitorID] = effect.destination;
		}
		beforeResources = resources(*callback);
		revision = contextState.revision;
		consume(contextState, callback->getDate(Date::DAY), decision, "engine");
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(blocked)
	{
		contextualResult(resultNode("rejected", false), decision);
		return true;
	}
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		const auto * town = activeTown();
		const auto * visitor = town->getVisitingHero();
		const auto * garrison = town->getGarrisonHero();
		if(town->id.getNum() != townID || contextState.revision != revision
			|| (visitor ? visitor->id.getNum() : -1) != (accepted ? garrisonID : visitorID)
			|| (garrison ? garrison->id.getNum() : -1) != (accepted ? visitorID : garrisonID))
			throw std::runtime_error("Garrison acknowledgement does not match exact hero slots");
		Armies after{{townID, snapshot(*town)}};
		if(visitor) after.emplace(visitor->id.getNum(), snapshot(*visitor));
		if(garrison) after.emplace(garrison->id.getNum(), snapshot(*garrison));
		if(after != (accepted ? expected : before) || resources(*callback) != beforeResources
			|| (visitor && visitor->isGarrisoned()) || (garrison && !garrison->isGarrisoned()))
			throw std::runtime_error("Garrison acknowledgement does not match exact native army and resource effects");
	}
	contextualResult(resultNode(accepted ? "garrison_swapped" : "rejected", accepted), decision);
	return true;
}
