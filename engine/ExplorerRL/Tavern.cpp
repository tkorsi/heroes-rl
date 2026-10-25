#include "StdInc.h"
#include "ExplorerRL.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/IGameSettings.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"

#include <regex>
#include <set>

namespace
{
// The real window has two portrait positions. Only returned nonempty cards
// are routed; the callback's unrestricted pool is never inspected.
std::vector<const CGHeroInstance *> offeredHeroes(const CCallback & callback, const CGTownInstance & town)
{
	if(callback.getPlayerID() != town.getOwner() || !town.hasBuilt(BuildingID::TAVERN))
		throw std::runtime_error("Tavern offers require an owned town with a tavern");
	auto offers = callback.getAvailableHeroes(&town);
	if(offers.size() > 2)
		throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: tavern offer layout");
	offers.erase(std::remove(offers.begin(), offers.end(), nullptr), offers.end());
	return offers;
}

const char * hireStatus(const CCallback & callback, const CGTownInstance & town, bool hasOffer)
{
	// CTavernWindow's visible recruit-button availability, in native order.
	if(callback.getResourceAmount(EGameResID::GOLD) < GameConstants::HERO_GOLD_COST) return "no_gold";
	if(callback.howManyHeroes(true) >= callback.getSettings().getInteger(EGameSettings::HEROES_PER_PLAYER_TOTAL_CAP)) return "total_limit";
	if(callback.howManyHeroes(false) >= callback.getSettings().getInteger(EGameSettings::HEROES_PER_PLAYER_ON_MAP_CAP)) return "map_limit";
	if(town.getVisitingHero()) return "visitor_present";
	return hasOffer ? "allowed" : "no_offer";
}

JsonNode tavernResult(const char * status, bool accepted)
{
	JsonNode result;
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	for(const char * field : {"position", "movement", "destination_reached"}) result[field] = JsonNode();
	return result;
}

using Army = std::map<int, std::pair<int, int64_t>>;
Army snapshotArmy(const CArmedInstance & army)
{
	Army result;
	for(const auto & [slot, stack] : army.Slots())
		result.emplace(slot.getNum(), std::pair(stack->getCreatureID().getNum(), stack->getCount()));
	return result;
}
}

JsonNode ExplorerRL::tavernPanel(const CGTownInstance & town)
{
	// This function is called only while this player's tavern is open. A
	// portrait does not reveal its army, skills, mana or hidden map history.
	if(!heroRecruitment || contextState.context != "town" || contextState.panel != "tavern")
		throw std::runtime_error("Tavern details require their explicit opened context");
	const auto offers = offeredHeroes(*callback, town);
	JsonNode result;
	result["offers"].Vector();
	result["selected_offer_ref"] = JsonNode();
	result["selected_detail"] = JsonNode();
	result["gold_cost"].Integer() = GameConstants::HERO_GOLD_COST;
	result["recruit_status"].String() = hireStatus(*callback, town, !offers.empty());
	static const std::regex name("hero\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name");
	for(const auto * hero : offers)
	{
		const auto nameID = hero->getNameTextID();
		if(!std::regex_match(nameID, name) || hero->getIconIndex() < 0)
			throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: tavern hero identity");
		auto [handle, inserted] = contextState.tavernHandles.try_emplace(hero->getHeroTypeID().getNum());
		if(inserted) handle->second = "hero-offer-" + std::to_string(contextState.tavernHandles.size() - 1);
		JsonNode card;
		card["ref"].String() = handle->second;
		card["portrait"].Integer() = hero->getIconIndex();
		card["name_text_id"].String() = nameID;
		result["offers"].Vector().push_back(std::move(card));
	}
	if(!offers.empty())
	{
		if(contextState.selectedTavernIndex >= offers.size())
			throw std::runtime_error("Tavern selected card is no longer displayed");
		const auto * selected = offers[contextState.selectedTavernIndex];
		result["selected_offer_ref"].String() = contextState.tavernHandles.at(selected->getHeroTypeID().getNum());
		auto & detail = result["selected_detail"];
		detail["level"].Integer() = selected->level;
		detail["hero_class"].Integer() = selected->getHeroClassID().getNum();
		int artifacts = static_cast<int>(selected->artifactsWorn.size() + selected->artifactsInBackpack.size());
		// Exactly the native selected-card description: do not name artifacts.
		for(int slot = 13; slot <= 17; ++slot)
			if(selected->artifactsWorn.count(ArtifactPosition(slot))) --artifacts;
		if(artifacts < 0) throw std::runtime_error("Invalid tavern artifact count");
		detail["artifact_count"].Integer() = artifacts;
	}
	return result;
}

bool ExplorerRL::executeTavernAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool opening = kind == "open_panel";
	const bool buying = kind == "hire_hero";
	const bool selecting = kind == "select_hero";
	if(!heroRecruitment || (!opening && !buying && !selecting && kind != "back")
		|| !action["subject"].isString() || !action["x"].isNull() || !action["y"].isNull()
		|| !action["count"].isNull() || !action["destination"].isNull()
		|| (opening ? action["panel"].String() != "tavern" : !action["panel"].isNull())
		|| ((buying || selecting) ? !action["selection"].isString() : !action["selection"].isNull()))
		throw std::runtime_error("Incorrect tavern action parameters");
	const auto subject = action["subject"].String();
	const auto selection = action["selection"].isString() ? action["selection"].String() : std::string();
	ObjectInstanceID townID(-1);
	HeroTypeID hiredType(-1);
	std::array<int64_t, 7> resources{};
	std::set<int> beforeHeroes;
	Army offeredArmy, townArmy;
	ObjectInstanceID upperArmyID(-1);
	const auto activeTown = [&]() -> const CGTownInstance *
	{
		if(contextState.context != "town" || contextState.subject != subject)
			throw std::runtime_error("Tavern action does not match its active town");
		const auto handle = contextState.subjects.find(subject);
		if(handle == contextState.subjects.end()) throw std::runtime_error("Unknown tavern town");
		for(const auto * town : callback->getTownsInfo())
			if(town->id.getNum() == handle->second && town->getOwner() == playerID) return town;
		throw std::runtime_error("Tavern town is no longer owned");
	};
	const auto validate = [&](const CGTownInstance & town)
	{
		if(!town.hasBuilt(BuildingID::TAVERN)) throw std::runtime_error("Town has no tavern");
		if(opening)
		{
			if(contextState.panel != "overview") throw std::runtime_error("Tavern entry requires town overview");
			const auto & controls = contextState.lastObservation["window"]["town_panel"]["controls"].Vector();
			if(std::none_of(controls.begin(), controls.end(), [](const auto & control) { return control["panel"].String() == "tavern"; }))
				throw std::runtime_error("Tavern control was not displayed");
			return;
		}
		if(contextState.panel != "tavern") throw std::runtime_error("Hero selection requires the opened tavern");
		if(kind == "back") return;
		const auto & panel = contextState.lastObservation["window"]["town_panel"]["tavern"];
		const auto & offers = panel["offers"].Vector();
		if(std::none_of(offers.begin(), offers.end(), [&](const auto & offer) { return offer["ref"].String() == selection; }))
			throw std::runtime_error("Hero offer was not displayed");
		if(buying && (panel["selected_offer_ref"].String() != selection || panel["recruit_status"].String() != "allowed"))
			throw std::runtime_error("Hero recruit control is disabled or not selected");
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
	const auto changePanel = [&](const char * panel)
	{
		contextState.panel = panel;
		contextState.selection.clear();
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
				contextState.selectedTavernIndex = 0;
				changePanel("tavern");
			}
			else if(kind == "back") changePanel("overview");
			else
			{
				const auto & offers = contextState.lastObservation["window"]["town_panel"]["tavern"]["offers"].Vector();
				const auto found = std::find_if(offers.begin(), offers.end(), [&](const auto & offer) { return offer["ref"].String() == selection; });
				contextState.selectedTavernIndex = static_cast<size_t>(std::distance(offers.begin(), found));
				changePanel("tavern");
			}
		}
	}
	if(!buying)
	{
		contextualResult(tavernResult(opening ? "opened" : selecting ? "selected" : "closed", true), decision);
		return true;
	}

	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto * town = activeTown();
		const auto offers = offeredHeroes(*callback, *town);
		if(town->id != townID || contextState.selectedTavernIndex >= offers.size()
			|| offers[contextState.selectedTavernIndex]->getHeroTypeID() != hiredType)
			throw std::runtime_error("Revalidated hero offer changed before issue");
		callback->recruitHero(town, offers[contextState.selectedTavernIndex], HeroTypeID::NONE);
		// CTavernWindow::recruitb closes before incoming state mutations. A
		// replacement card is not refreshed through this now-closed window.
		changePanel("overview");
	}, true, decision, &issued, [&]
	{
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto * town = activeTown();
		validate(*town);
		const auto offers = offeredHeroes(*callback, *town);
		if(contextState.selectedTavernIndex >= offers.size()
			|| std::string(hireStatus(*callback, *town, !offers.empty())) != "allowed")
			throw std::runtime_error("Hero recruit availability changed before issue");
		const auto * hero = offers[contextState.selectedTavernIndex];
		hiredType = hero->getHeroTypeID();
		if(contextState.tavernHandles.at(hiredType.getNum()) != selection)
			throw std::runtime_error("Hero recruit selection changed before issue");
		townID = town->id;
		offeredArmy = snapshotArmy(*hero); // receipt audit only; never a policy field
		upperArmyID = town->getUpperArmy()->id;
		townArmy = snapshotArmy(*town->getUpperArmy());
		for(const auto * owned : callback->getHeroesInfo())
			if(owned->getOwner() == playerID) beforeHeroes.insert(owned->id.getNum());
		for(int resource = 0; resource < 7; ++resource) resources[resource] = callback->getResourceAmount(GameResID(resource));
		consume("engine");
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		const auto * town = activeTown();
		std::set<int> afterHeroes;
		const CGHeroInstance * added = nullptr;
		for(const auto * hero : callback->getHeroesInfo())
			if(hero->getOwner() == playerID)
			{
				afterHeroes.insert(hero->id.getNum());
				if(!beforeHeroes.count(hero->id.getNum()))
				{
					if(added) throw std::runtime_error("Hero recruit created multiple owned heroes");
					added = hero;
				}
			}
		if(accepted)
		{
			if(!added || added->getHeroTypeID() != hiredType || snapshotArmy(*added) != offeredArmy
				|| town->getVisitingHero() != added || added->visitablePos() != town->visitablePos())
				throw std::runtime_error("Hero recruit acknowledgement does not match its selected hero and army");
			afterHeroes.erase(added->id.getNum());
		}
		else if(added || town->getVisitingHero())
			throw std::runtime_error("Rejected hero recruit mutated the visitor");
		if(afterHeroes != beforeHeroes || town->id != townID || town->getUpperArmy()->id != upperArmyID
			|| snapshotArmy(*town->getUpperArmy()) != townArmy)
			throw std::runtime_error("Hero recruit acknowledgement changed existing heroes or town army");
		for(int resource = 0; resource < 7; ++resource)
			if(resources[resource] - callback->getResourceAmount(GameResID(resource))
				!= (accepted && resource == EGameResID::GOLD ? GameConstants::HERO_GOLD_COST : 0))
				throw std::runtime_error("Hero recruit acknowledgement does not match exact resource effects");
	}
	contextualResult(tavernResult(accepted ? "hired" : "rejected", accepted), decision);
	return true;
}
