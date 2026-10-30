#include "StdInc.h"
#include "ExplorerRL.h"
#include "ContextualEvidence.h"
#include "KnowledgePlanner.h"
#include "Notice.h"
#include "VisibleWorld.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/entities/artifact/ArtifactUtils.h"
#include "../../lib/entities/artifact/CArtifactInstance.h"
#include "../../lib/entities/building/CBuilding.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"
#include "../../lib/mapping/TerrainTile.h"
#include "../../lib/networkPacks/ArtifactLocation.h"

#include <algorithm>
#include <limits>
#include <set>

namespace
{
JsonNode coordinates(const int3 & position)
{
	JsonNode node;
	for(const int value : {position.x, position.y, position.z})
		node.Vector().emplace_back(value);
	return node;
}

std::string relativeOwner(PlayerColor owner, PlayerColor player)
{
	return owner == player ? "self" : owner.isValidPlayer() ? "opponent" : "neutral";
}

JsonNode army(const CArmedInstance & object)
{
	JsonNode node;
	node.Vector();
	for(const auto & [slot, stack] : object.Slots())
	{
		JsonNode item;
		item["slot"].Integer() = slot.getNum();
		item["creature"].Integer() = stack->getCreatureID().getNum();
		item["count"].Integer() = stack->getCount();
		node.Vector().push_back(std::move(item));
	}
	return node;
}

JsonNode ownArmyDetails(const CArmedInstance & object, PlayerColor player, const std::string & subject, const char * role)
{
	// This opt-in view may read only the player's own callback-provided armies.
	// It intentionally has no path through visible enemy/neutral object handles.
	if(object.getOwner() != player)
		throw std::runtime_error("Passive army information requires an owned army");
	JsonNode result;
	result["subject"].String() = subject;
	result["role"].String() = role;
	result["slots"].Vector();
	for(const auto & [slot, stack] : object.Slots())
	{
		if(!slot.validSlot() || !stack || !stack->getCreature() || stack->getCount() <= 0
			|| stack->getCount() > std::numeric_limits<int32_t>::max())
			throw std::runtime_error("Unsupported owned army detail slot");
		// Match CCreatureWindow's adventure-map popup convention: attack,
		// defense and damage use ranged mode for a shooter with shots. These
		// are current stack/hero bonuses, not the creature's unmodified base.
		const int shots = stack->valOfBonuses(BonusType::SHOTS);
		const bool shooter = stack->hasBonusOfType(BonusType::SHOOTER) && shots != 0;
		JsonNode item;
		item["slot"].Integer() = slot.getNum();
		item["creature"].Integer() = stack->getCreatureID().getNum();
		item["count"].Integer() = stack->getCount();
		item["name_text_id"].String() = stack->getCreature()->getNamePluralTextID();
		item["attack"].Integer() = stack->getAttack(shooter);
		item["defense"].Integer() = stack->getDefense(shooter);
		item["damage_min"].Integer() = stack->getMinDamage(shooter);
		item["damage_max"].Integer() = stack->getMaxDamage(shooter);
		item["health"].Integer() = stack->getMaxHealth();
		item["speed"].Integer() = stack->getMovementRange();
		item["shots"].Integer() = shooter ? shots : 0;
		item["casts"].Integer() = stack->valOfBonuses(BonusType::CASTS);
		// MoraleLuckBox suppresses these values even when a MAX_* bonus is
		// present, so apply its displayed immunity override explicitly.
		item["morale"].Integer() = stack->unaffectedByMorale() ? 0 : stack->moraleVal();
		item["luck"].Integer() = stack->hasBonusOfType(BonusType::NO_LUCK) ? 0 : stack->luckVal();
		item["shooter"].Bool() = shooter;
		item["flying"].Bool() = stack->hasBonusOfType(BonusType::FLYING);
		item["undead"].Bool() = stack->hasBonusOfType(BonusType::UNDEAD);
		result["slots"].Vector().push_back(std::move(item));
	}
	std::sort(result["slots"].Vector().begin(), result["slots"].Vector().end(), [](const JsonNode & first, const JsonNode & second)
	{
		return first["slot"].Integer() < second["slot"].Integer();
	});
	return result;
}

void fields(const JsonNode & value, const std::set<std::string> & allowed, const std::set<std::string> & required)
{
	if(!value.isStruct())
		throw std::runtime_error("Contextual response requires an object");
	for(const auto & [name, unused] : value.Struct())
		if(!allowed.count(name))
			throw std::runtime_error("Unknown contextual response field");
	for(const auto & name : required)
		if(!value.Struct().count(name))
			throw std::runtime_error("Missing contextual response field");
}

bool integer(const JsonNode & value)
{
	return value.getType() == JsonNode::JsonType::DATA_INTEGER;
}

JsonNode resultNode(const char * status, bool accepted)
{
	JsonNode result;
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	result["position"] = JsonNode();
	result["movement"] = JsonNode();
	result["destination_reached"] = JsonNode();
	return result;
}
}

namespace explorer_rl
{
std::string ContextualState::handle(int engineID)
{
	if(const auto found = handles.find(engineID); found != handles.end())
		return found->second;
	const std::string value = "object-" + std::to_string(handles.size());
	handles.emplace(engineID, value);
	subjects.emplace(value, engineID);
	return value;
}

JsonNode ContextualState::contextJson() const
{
	JsonNode node;
	node["kind"].String() = context;
	node["subject"] = context == "map" ? JsonNode() : JsonNode(subject);
	if((context == "town" && !panel.empty()) || context == "stack" || context == "exchange" || context == "dialog" || context == "capability")
	{
		node["panel"] = panel.empty() ? JsonNode() : JsonNode(panel);
		node["selection"] = selection.empty() ? JsonNode() : JsonNode(selection);
	}
	return node;
}

JsonNode ContextualState::record(const char * kind, int day) const
{
	JsonNode node;
	node["kind"].String() = kind;
	node["day"].Integer() = day;
	return node;
}

void ContextualState::supersede(int day, const char * reason)
{
	if(!decision)
		return;
	auto node = record("superseded", day);
	node["decision_id"].Integer() = *decision;
	node["reason"].String() = reason;
	records.push_back(std::move(node));
	decision.reset();
	executing = false;
	issueArmed = false;
	routeInteractions.reset();
	executingHero.reset();
	requests.clear();
}
}

JsonNode ExplorerRL::contextualObservation()
{
	// Caller holds a stable engine view and stateMutex. Historical observations
	// still require opened windows for armies; only the explicit owned-army
	// extension adds passive composition and popup statistics on the map path.
	const auto size = callback->getMapSize();
	if(size.z != 1 || size.x <= 0 || size.y <= 0)
		throw std::runtime_error("Contextual interface requires one map level");
	JsonNode observation;
	observation["size"] = coordinates(size);
	for(const char * key : {"tiles", "objects", "resources", "heroes", "towns"})
		observation[key].Vector();
	observation["window"] = JsonNode();
	if(ownArmyInformation)
	{
		observation["army_information"].String() = explorer_rl::ContextualEvidence::OWN_ARMY_INFORMATION;
		observation["own_armies"].Vector();
	}
	const CGHeroInstance * openHero = nullptr;
	const CGTownInstance * openTown = nullptr;
	for(const auto * hero : callback->getHeroesInfo())
	{
		if(hero->getOwner() != playerID)
			continue;
		JsonNode item;
		const auto handle = contextState.handle(hero->id.getNum());
		item["handle"].String() = handle;
		item["position"] = coordinates(hero->visitablePos());
		item["movement"].Integer() = hero->movementPointsRemaining();
		item["movement_max"].Integer() = hero->movementPointsLimit();
		if(armyManagement) item["in_garrison"].Bool() = hero->isGarrisoned();
		observation["heroes"].Vector().push_back(std::move(item));
		if(ownArmyInformation)
			observation["own_armies"].Vector().push_back(ownArmyDetails(*hero, playerID, handle, "hero"));
		if(contextState.context == "hero" && contextState.subject == handle)
			openHero = hero;
	}
	for(const auto * town : callback->getTownsInfo())
	{
		if(town->getOwner() != playerID)
			continue;
		JsonNode item;
		const auto handle = contextState.handle(town->id.getNum());
		item["handle"].String() = handle;
		item["position"] = coordinates(town->visitablePos());
		observation["towns"].Vector().push_back(std::move(item));
		// A garrison hero was already emitted with the owned heroes. Reading
		// the town itself, rather than getUpperArmy(), avoids duplicating it.
		if(ownArmyInformation)
			observation["own_armies"].Vector().push_back(ownArmyDetails(*town, playerID, handle, "town_garrison"));
		if(contextState.context == "town" && contextState.subject == handle)
			openTown = town;
	}
	if(ownArmyInformation)
		std::sort(observation["own_armies"].Vector().begin(), observation["own_armies"].Vector().end(), [](const JsonNode & first, const JsonNode & second)
		{
			return first["subject"].String() < second["subject"].String();
		});
	for(int resource = 0; resource < 7; ++resource)
		observation["resources"].Vector().emplace_back(callback->getResourceAmount(GameResID(resource)));
	std::set<int> seen;
	for(int y = 0; y < size.y; ++y)
		for(int x = 0; x < size.x; ++x)
		{
			const int3 position(x, y, 0);
			if(!callback->isVisible(position))
				continue;
			const auto * tile = callback->getTile(position, false);
			if(!tile)
				throw std::runtime_error("Visible contextual tile is unavailable");
			const auto facts = explorer_rl::visibleTileFacts(*callback, *tile);
			JsonNode item;
			item["position"] = coordinates(position);
			item["terrain"].Integer() = facts.terrain;
			item["road"].Integer() = tile->getRoadID().getNum();
			item["blocked"].Bool() = facts.blocked;
			item["visitable"].Bool() = facts.visitable;
			observation["tiles"].Vector().push_back(std::move(item));
			for(const auto id : tile->visitableObjects)
			{
				const auto * object = explorer_rl::playerVisibleObject(*callback, id);
				if(!object || !seen.insert(id.getNum()).second)
					continue;
				JsonNode entry;
				entry["handle"].String() = contextState.handle(id.getNum());
				// This is a visible visitable tile, never an unseen footprint anchor.
				entry["position"] = coordinates(position);
				entry["kind"].Integer() = object->ID.getNum();
				entry["subtype"].Integer() = object->subID.getNum();
				entry["owner"].String() = relativeOwner(object->getOwner(), playerID);
				observation["objects"].Vector().push_back(std::move(entry));
			}
		}
	const bool dialogModal = adventureDialogs && contextState.context == "dialog";
	const bool capabilityModal = expandedInterface && contextState.context == "capability";
	const bool armyModal = armyManagement && (contextState.context == "exchange" || contextState.context == "stack");
	if(contextState.context != "map" && !openHero && !openTown && !armyModal && !dialogModal && !capabilityModal)
	{
		const auto day = callback->getDate(Date::DAY);
		contextState.supersede(day, "subject_unavailable");
		contextState.context = "map";
		contextState.subject.clear();
		contextState.panel.clear();
		contextState.selection.clear();
		auto record = contextState.record("context_invalidated", day);
		record["context"] = contextState.contextJson();
		record["revision"].Integer() = ++contextState.revision;
		record["reason"].String() = "subject_unavailable";
		contextState.records.push_back(std::move(record));
	}
	if(openHero || openTown || armyModal || dialogModal || capabilityModal)
	{
		JsonNode window;
		window["context"] = contextState.contextJson();
		for(const char * key : {"army", "primary_skills", "mana", "level", "buildings", "faction"})
			window[key] = JsonNode();
		if(openHero)
		{
			window["army"] = army(*openHero);
			for(int skill = 0; skill < 4; ++skill)
				window["primary_skills"].Vector().emplace_back(openHero->getPrimSkillLevel(PrimarySkill(skill)));
			window["mana"].Integer() = openHero->mana;
			window["level"].Integer() = openHero->level;
		}
		else if(openTown)
		{
			if(townEconomy && contextState.panel != "overview")
				window["army"] = JsonNode();
			else
			{
				window["army"] = army(*openTown->getUpperArmy());
				window["faction"].Integer() = openTown->getFactionID().getNum();
			}
			if(heroRecruitment && contextState.panel == "tavern")
			{
				auto & panel = window["town_panel"];
				panel["kind"].String() = "tavern";
				panel["controls"].Vector();
				panel["buildings"].Vector();
				panel["building"] = JsonNode();
				panel["tavern"] = tavernPanel(*openTown);
			}
			else if(townEconomy)
				window["town_panel"] = explorer_rl::townPanel(*callback, *openTown,
					contextState.panel, contextState.selection, contextState.economy);
			if(heroRecruitment && contextState.panel == "overview" && openTown->hasBuilt(BuildingID::TAVERN))
			{
				JsonNode control;
				control["panel"].String() = "tavern";
				control["ref"].String() = "panel-1";
				window["town_panel"]["controls"].Vector().push_back(std::move(control));
			}
			if(armyTransfer && contextState.panel == "overview")
				window["town_panel"]["armies"] = townArmyExchange(*openTown);
		}
		if(armyManagement)
		{
			window["armies"] = contextualArmyViews();
			window["stack_panel"] = contextState.context == "stack" ? stackPanel() : JsonNode();
		}
		if(adventureDialogs) window["dialog_panel"] = dialogModal ? adventureDialogPanel() : JsonNode();
		observation["window"] = std::move(window);
	}
	if(expandedInterface && !observation["window"].isNull())
	{
		observation["window"]["capability_controls"] = capabilityControls();
		observation["window"]["capability_panel"] = capabilityModal ? capabilityPanel() : JsonNode();
		if(capabilityModal && contextState.panel == "spellbook")
			observation["window"]["mana"].Integer() = callback->getHero(ObjectInstanceID(contextState.subjects.at(contextState.subject)))->mana;
	}
	return observation;
}

void ExplorerRL::captureContextual()
{
	std::lock_guard lock(stateMutex);
	captureContextualLocked();
}

void ExplorerRL::captureContextualLocked()
{
	const int day = callback->getDate(Date::DAY);
	if(day < 1)
		return;
	auto observation = contextualObservation();
	if(observation == contextState.lastObservation && day == contextState.lastDay)
		return;
	if(!contextState.executing)
		contextState.supersede(day, "visible_state_changed");
	if(!contextState.lastObservation.isNull())
		for(const auto & previous : contextState.lastObservation["heroes"].Vector())
		{
			const auto & heroes = observation["heroes"].Vector();
			if(std::none_of(heroes.begin(), heroes.end(), [&](const auto & hero) { return hero["handle"] == previous["handle"]; }))
			{
				auto lost = contextState.record("event", day);
				lost["event"]["kind"].String() = "hero_lost";
				lost["event"]["subject"] = previous["handle"];
				contextState.records.push_back(std::move(lost));
			}
		}
	auto record = contextState.record("view", day);
	record["observation"] = observation;
	contextState.records.push_back(std::move(record));
	contextState.lastObservation = std::move(observation);
	contextState.lastDay = day;
}

JsonNode ExplorerRL::takeContextualRecords()
{
	auto envelope = message("contextual");
	std::lock_guard lock(stateMutex);
	envelope["records"].Vector() = std::move(contextState.records);
	contextState.records.clear();
	return envelope;
}

void ExplorerRL::contextualEvent(JsonNode event)
{
	const int day = callback->getDate(Date::DAY);
	std::lock_guard lock(stateMutex);
	if(stopping)
		return;
	if(!contextState.executing)
		contextState.supersede(day, "visible_event");
	const auto & kind = event["kind"].String();
	if(kind == "visit" || kind == "battle_start" || kind == "battle_complete" || kind == "notice")
		++contextState.interactions;
	auto record = contextState.record("event", day);
	record["event"] = std::move(event);
	contextState.records.push_back(std::move(record));
}

void ExplorerRL::contextualTerminal(const char * outcome)
{
	const int day = callback->getDate(Date::DAY);
	{
		std::lock_guard lock(stateMutex);
		contextState.supersede(day, "terminal");
		contextState.ownHeroExchange.reset();
		contextState.adventureDialog.reset();
		contextState.capabilityDialog.reset();
		contextState.executingHero.reset();
		contextState.stackParentContext = JsonNode();
		contextState.selectedUpgradeRef.clear();
		if(contextState.context != "map")
		{
			contextState.context = "map";
			contextState.subject.clear();
			contextState.panel.clear();
			contextState.selection.clear();
			auto record = contextState.record("context_invalidated", day);
			record["context"] = contextState.contextJson();
			record["revision"].Integer() = ++contextState.revision;
			record["reason"].String() = "terminal";
			contextState.records.push_back(std::move(record));
		}
	}
	captureContextual();
	{
		std::lock_guard lock(stateMutex);
		auto record = contextState.record("terminal", day);
		record["outcome"].String() = outcome;
		contextState.records.push_back(std::move(record));
	}
	writeMessage(takeContextualRecords());
}

void ExplorerRL::contextualResult(JsonNode result, int64_t decision)
{
	buildAndSend([&]
	{
		const auto gameLock = readGameState();
		captureContextual();
		{
			std::lock_guard lock(stateMutex);
			if(contextState.decision == decision && !terminal)
			{
				auto record = contextState.record("result", callback->getDate(Date::DAY));
				record["decision_id"].Integer() = decision;
				record["result"] = std::move(result);
				contextState.records.push_back(std::move(record));
				contextState.decision.reset();
				contextState.executing = false;
				contextState.executingHero.reset();
				contextState.routeInteractions.reset();
			}
		}
		return takeContextualRecords();
	});
}

void ExplorerRL::runContextualTurn(QueryID queryID, bool modalOnly)
{
	if(queryID != QueryID(-1) && !executeRequest([this, queryID] { callback->selectionMade(0, queryID); }))
	{
		if(!ended())
			throw std::runtime_error("Engine rejected turn acknowledgement");
		return;
	}
	while(!ended())
	{
		if(modalOnly)
		{
			std::lock_guard lock(stateMutex);
			if(!contextState.capabilityDialog) return;
		}
		int64_t decision = -1;
		int64_t revision = -1;
		if(!buildAndSend([&]
		{
			const auto gameLock = readGameState();
			{
				std::lock_guard lock(stateMutex);
				installOwnHeroExchange();
				installAdventureDialog();
				installCapabilityDialog();
			}
			captureContextual();
			{
				std::lock_guard lock(stateMutex);
				if(contextState.decision)
					throw std::logic_error("Previous contextual action has not completed");
				decision = contextState.nextDecision++;
				contextState.decision = decision;
				revision = ++contextState.revision;
				auto record = contextState.record("decision", callback->getDate(Date::DAY));
				record["decision_id"].Integer() = decision;
				record["revision"].Integer() = revision;
				record["observation"] = contextState.lastObservation;
				contextState.records.push_back(std::move(record));
			}
			return takeContextualRecords();
		}))
			return;
		const auto response = receiveMessage();
		if(ended())
			return;
		fields(response, {"protocol_version", "type", "episode", "player", "decision_id", "revision", "action"},
			{"protocol_version", "type", "episode", "player", "decision_id", "revision", "action"});
		if(!integer(response["protocol_version"]) || response["protocol_version"].Integer() != (expandedInterface ? 10 : adventureDialogs ? 9 : armyManagement ? 8 : heroRecruitment ? 7 : armyTransfer ? 6 : townEconomy ? 5 : 4)
			|| response["type"].String() != "context_action" || response["episode"].String() != episode
			|| !integer(response["player"]) || response["player"].Integer() != playerID.getNum()
			|| !integer(response["decision_id"]) || response["decision_id"].Integer() != decision
			|| !integer(response["revision"]) || response["revision"].Integer() != revision)
			throw std::runtime_error("Contextual response identity or revision mismatch");
		const bool consumed = executeContextualAction(response["action"], decision);
		if(consumed && response["action"]["kind"].String() == "end_turn")
			return;
	}
}

bool ExplorerRL::executeTownAction(const JsonNode & action, int64_t decision)
{
	const auto kind = action["kind"].String();
	const bool constructing = kind == "construct";
	if(!action["subject"].isString() || !action["x"].isNull() || !action["y"].isNull()
		|| !action["count"].isNull() || !action["destination"].isNull())
		throw std::runtime_error("Incorrect town action parameters");
	if(kind == "open_panel")
	{
		if(!action["panel"].isString() || action["panel"].String() != "hall" || !action["selection"].isNull())
			throw std::runtime_error("Unsupported town panel selection");
	}
	else if(!action["panel"].isNull() || (kind == "back" ? !action["selection"].isNull() : !action["selection"].isString()))
		throw std::runtime_error("Incorrect building selection");
	const auto subject = action["subject"].String();
	const auto selection = action["selection"].isString() ? action["selection"].String() : std::string();
	ObjectInstanceID townID(-1);
	BuildingID buildingID(-1);
	std::array<int64_t, 7> beforeResources{}, price{};
	int beforeBuilt = 0;

	// These helpers run only with both stable game state and stateMutex held.
	const auto activeTown = [&]() -> const CGTownInstance *
	{
		if(contextState.context != "town" || contextState.subject != subject)
			throw std::runtime_error("Town action does not match the active context");
		const auto handle = contextState.subjects.find(subject);
		if(handle == contextState.subjects.end())
			throw std::runtime_error("Unknown town subject");
		for(const auto * town : callback->getTownsInfo())
			if(town->id.getNum() == handle->second && town->getOwner() == playerID)
				return town;
		throw std::runtime_error("Active town is no longer owned");
	};
	const auto validate = [&](const CGTownInstance & town)
	{
		if(kind == "open_panel")
		{
			if(contextState.panel != "overview")
				throw std::runtime_error("Hall entry requires town overview");
		}
		else if(kind == "open_building")
		{
			if(contextState.panel != "hall")
				throw std::runtime_error("Building selection requires hall");
			const auto & choices = contextState.lastObservation["window"]["town_panel"]["buildings"].Vector();
			if(std::none_of(choices.begin(), choices.end(), [&](const auto & entry) { return entry["ref"].String() == selection; }))
				throw std::runtime_error("Building was not displayed in this hall");
			contextState.economy.resolveBuilding(town, selection);
		}
		else if(kind == "back")
		{
			if(contextState.panel != "hall" && contextState.panel != "building")
				throw std::runtime_error("Back requires a supported town overlay");
		}
		else if(constructing)
		{
			if(contextState.panel != "building" || contextState.selection != selection)
				throw std::runtime_error("Construction requires the selected building detail");
			const auto id = contextState.economy.resolveBuilding(town, selection);
			const auto & building = *town.getTown()->buildings.at(id);
			if(building.mode != CBuilding::BUILD_NORMAL
				|| callback->canBuildStructure(&town, id) != EBuildingState::ALLOWED)
				throw std::runtime_error("Construction control is disabled");
		}
	};
	const auto consume = [&]()
	{
		contextState.executing = true;
		contextState.commands = 0;
		contextState.routeInteractions.reset();
		auto record = contextState.record("consumed", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["consumer"].String() = constructing ? "engine" : "adapter";
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
		if(!constructing)
		{
			consume();
			if(kind == "open_panel")
			{
				if(explorer_rl::hallHasGrailPrompt(*town))
					throw std::runtime_error("Unsupported scenario interaction: unsupported_interaction: hall Grail prompt");
				changePanel("hall", "");
			}
			else if(kind == "open_building") changePanel("building", selection);
			else changePanel(contextState.panel == "building" ? "hall" : "overview", "");
		}
	}
	if(!constructing)
	{
		contextualResult(resultNode(kind == "back" ? "closed" : "opened", true), decision);
		return true;
	}

	bool issued = false;
	const bool accepted = executeRequest([&]
	{
		const auto * town = activeTown();
		if(town->id != townID || !callback->buildBuilding(town, buildingID))
			throw std::runtime_error("Revalidated construction did not issue a native request");
		// CBuildWindow::buyFunc pops detail and hall immediately after sending,
		// even if the later server acknowledgement rejects the request. Holding
		// the read/state locks ensures no incoming change refreshes closed detail.
		changePanel("overview", "");
	}, true, decision, &issued, [&]
	{
		// A changed held decision is superseded before consumption, under the
		// same read/state locks that protect the actual send and requestSent.
		captureContextualLocked();
		if(contextState.decision != decision) return false;
		const auto * town = activeTown();
		validate(*town);
		townID = town->id;
		buildingID = contextState.economy.resolveBuilding(*town, selection);
		const auto & cost = town->getTown()->buildings.at(buildingID)->resources;
		beforeBuilt = town->built;
		for(int resource = 0; resource < 7; ++resource)
		{
			beforeResources[resource] = callback->getResourceAmount(GameResID(resource));
			price[resource] = cost[resource];
		}
		consume();
		contextState.issueArmed = true;
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	{
		const auto gameLock = readGameState();
		std::lock_guard lock(stateMutex);
		const auto * town = activeTown();
		if(town->id != townID || town->hasBuilt(buildingID) != accepted
			|| town->built != beforeBuilt + (accepted ? 1 : 0))
			throw std::runtime_error("Construction acknowledgement does not match the observed building effect");
		for(int resource = 0; resource < 7; ++resource)
			if(beforeResources[resource] - callback->getResourceAmount(GameResID(resource)) != (accepted ? price[resource] : 0))
				throw std::runtime_error("Construction acknowledgement does not match the observed resource effect");
	}
	contextualResult(resultNode(accepted ? "built" : "rejected", accepted), decision);
	return true;
}

bool ExplorerRL::executeContextualAction(const JsonNode & action, int64_t decision)
{
	fields(action, townEconomy ? std::set<std::string>{"schema", "kind", "subject", "x", "y", "panel", "selection", "count", "destination"}
		: std::set<std::string>{"schema", "kind", "subject", "x", "y"}, {"schema", "kind"});
	if(action["schema"].String() != (expandedInterface ? "heroes-context-actions-v8" : adventureDialogs ? "heroes-context-actions-v7" : armyManagement ? "heroes-context-actions-v6" : heroRecruitment ? "heroes-context-actions-v5" : armyTransfer ? "heroes-context-actions-v4" : townEconomy ? "heroes-context-actions-v3" : "heroes-context-actions-v2") || !action["kind"].isString())
		throw std::runtime_error("Unsupported contextual action schema");
	const std::string kind = action["kind"].String();
	if(expandedInterface)
	{
		bool active;
		{ std::lock_guard lock(stateMutex); active = contextState.context == "capability"; }
		if(active || kind == "open_capability") return executeCapabilityAction(action, decision);
	}
	if(adventureDialogs)
	{
		bool activeDialog;
		{ std::lock_guard lock(stateMutex); activeDialog = contextState.context == "dialog"; }
		if(activeDialog || kind == "choose_dialog") return executeDialogAction(action, decision);
	}
	if(armyManagement)
	{
		bool stackBack = false, exchangeClose = false;
		{
			std::lock_guard lock(stateMutex);
			stackBack = kind == "back" && contextState.context == "stack";
			exchangeClose = kind == "close" && contextState.context == "exchange";
		}
		if(kind == "split_stack" || kind == "merge_stacks" || kind == "swap_stacks")
			return executeArmyAction(action, decision);
		if(kind == "open_stack" || kind == "select_upgrade" || kind == "upgrade_stack" || stackBack)
			return executeStackAction(action, decision);
		if(kind == "swap_garrison") return executeGarrisonAction(action, decision);
		if(exchangeClose) return executeExchangeClose(action, decision);
	}
	if(heroRecruitment)
	{
		bool tavernBack = false;
		if(kind == "back")
		{
			std::lock_guard lock(stateMutex);
			tavernBack = contextState.context == "town" && contextState.panel == "tavern";
		}
		if(kind == "select_hero" || kind == "hire_hero" || tavernBack
			|| (kind == "open_panel" && action["panel"].isString() && action["panel"].String() == "tavern"))
			return executeTavernAction(action, decision);
	}
	if(armyTransfer && kind == "transfer_stack")
		return executeTransferAction(action, decision);
	if(townEconomy)
	{
		bool recruitmentBack = false;
		if(kind == "back")
		{
			std::lock_guard lock(stateMutex);
			recruitmentBack = contextState.context == "town" && contextState.panel == "recruitment";
		}
		if(kind == "select_recruit" || kind == "recruit" || recruitmentBack
			|| (kind == "open_panel" && action["panel"].isString() && action["panel"].String() == "recruitment"))
			return executeRecruitmentAction(action, decision);
	}
	if(townEconomy && (kind == "open_panel" || kind == "open_building" || kind == "back" || kind == "construct"))
		return executeTownAction(action, decision);
	if(townEconomy)
		for(const char * key : {"panel", "selection", "count", "destination"})
			if(!action[key].isNull()) throw std::runtime_error("Unused town action parameter");
	const bool navigation = kind == "open_hero" || kind == "open_town" || kind == "close";
	if(!navigation && kind != "move_to" && kind != "end_turn")
		throw std::runtime_error("Unsupported contextual action");
	const bool needsSubject = kind == "open_hero" || kind == "open_town" || kind == "move_to";
	if(needsSubject ? !action["subject"].isString() : !action["subject"].isNull())
		throw std::runtime_error("Incorrect contextual subject");
	if(kind == "move_to" ? (!integer(action["x"]) || !integer(action["y"])) : (!action["x"].isNull() || !action["y"].isNull()))
		throw std::runtime_error("Incorrect contextual coordinates");
	ObjectInstanceID subject(-1);
	int3 destination(-1, -1, -1);
	int64_t interactions = 0;
	{
		const auto gameLock = readGameState();
		captureContextual();
		std::lock_guard lock(stateMutex);
		if(contextState.decision != decision || terminal || stopping)
			return false; // The held decision was explicitly superseded; never consume its late reply.
		if(!failure.empty())
			throw std::runtime_error(failure);
		if((contextState.context == "map") == (kind == "close"))
			throw std::runtime_error("Action is unavailable in the current context");
		if(contextState.context != "map" && kind != "close")
			throw std::runtime_error("Only closing is supported in a window");
		if(townEconomy && kind == "close" && contextState.context == "town" && contextState.panel != "overview")
			throw std::runtime_error("Town overlay requires back before close");
		if(needsSubject)
		{
			const auto & own = contextState.lastObservation[kind == "open_town" ? "towns" : "heroes"].Vector();
			if(std::none_of(own.begin(), own.end(), [&](const auto & item) { return item["handle"] == action["subject"]; }))
				throw std::runtime_error("Contextual subject was not offered");
			subject = ObjectInstanceID(contextState.subjects.at(action["subject"].String()));
			if(armyManagement && kind == "move_to" && callback->getHero(subject)->isGarrisoned())
				throw std::runtime_error("A garrisoned hero has no map movement control");
		}
		if(kind == "move_to")
		{
			const auto size = callback->getMapSize();
			if(action["x"].Integer() < 0 || action["x"].Integer() >= size.x || action["y"].Integer() < 0 || action["y"].Integer() >= size.y)
				throw std::runtime_error("Destination outside public map bounds");
			destination = int3(static_cast<int>(action["x"].Integer()), static_cast<int>(action["y"].Integer()), 0);
		}
		contextState.executing = true;
		contextState.commands = 0;
		interactions = contextState.interactions;
		contextState.executingHero = adventureDialogs && kind == "move_to" ? std::optional<int>(subject.getNum()) : std::nullopt;
		contextState.routeInteractions = kind == "move_to" ? std::optional<int64_t>(interactions) : std::nullopt;
		auto consumed = contextState.record("consumed", callback->getDate(Date::DAY));
		consumed["decision_id"].Integer() = decision;
		consumed["consumer"].String() = navigation ? "adapter" : "engine";
		contextState.records.push_back(std::move(consumed));
		if(navigation)
		{
			contextState.context = kind == "close" ? "map" : kind == "open_hero" ? "hero" : "town";
			contextState.subject = needsSubject ? action["subject"].String() : "";
			contextState.panel = townEconomy && kind == "open_town" ? "overview" : "";
			contextState.selection.clear();
			auto change = contextState.record("context", callback->getDate(Date::DAY));
			change["decision_id"].Integer() = decision;
			change["context"] = contextState.contextJson();
			change["revision"].Integer() = ++contextState.revision;
			contextState.records.push_back(std::move(change));
		}
	}
	if(navigation)
	{
		contextualResult(resultNode(kind == "close" ? "closed" : "opened", true), decision);
		return true;
	}
	if(kind == "end_turn")
	{
		{
			std::lock_guard lock(stateMutex);
			contextState.issueArmed = true;
		}
		bool issued = false;
		const bool accepted = executeRequest([this] { callback->endTurn(); }, true, decision, &issued);
		if(!issued || ended())
		{
			if(ended())
				return true;
			throw std::runtime_error("Contextual end-turn was not issued");
		}
		if(!accepted)
			throw std::runtime_error("Engine rejected contextual turn ending");
		JsonNode event;
		event["kind"].String() = "turn_end";
		{
			const auto gameLock = readGameState();
			contextualEvent(std::move(event));
		}
		contextualResult(resultNode("turn_ended", true), decision);
		return true;
	}
	std::vector<explorer_rl::RouteStep> steps;
	std::string status;
	int3 expected;
	{
		const auto gameLock = readGameState();
		const auto * hero = callback->getHero(subject);
		if(!hero || hero->getOwner() != playerID)
			status = "hero_unavailable";
		else
		{
			expected = hero->visitablePos();
			if(expected == destination)
				status = "reached";
			else if(hero->movementPointsRemaining() == 0)
				status = "movement_limit";
			else
			{
				const auto start = std::chrono::steady_clock::now();
				const auto plan = explorer_rl::planRoute(*callback, *hero, destination);
				if(std::chrono::steady_clock::now() - start > std::chrono::seconds(timeoutSeconds))
					throw std::runtime_error("Knowledge route planning timed out");
				steps = plan.steps;
				if(!plan.known)
					status = "no_route_known";
				else if(steps.empty())
					status = "movement_limit";
			}
		}
	}
	const auto routeStart = std::chrono::steady_clock::now();
	for(const auto & routeStep : steps)
	{
		const auto & step = routeStep.position;
		if(ended())
			return true;
		if(std::chrono::steady_clock::now() - routeStart > std::chrono::seconds(timeoutSeconds))
			throw std::runtime_error("Contextual route execution timed out");
		if(step.z != expected.z || std::abs(step.x - expected.x) > 1 || std::abs(step.y - expected.y) > 1 || step == expected)
			throw std::runtime_error("Knowledge route contains a non-adjacent command");
		{
			std::lock_guard lock(stateMutex);
			if(contextState.interactions != interactions)
			{
				status = contextState.commands > 0 ? "interaction" : "interrupted";
				break;
			}
			contextState.issueArmed = true;
		}
		bool issued = false;
		const bool accepted = executeRequest([&, this]
		{
			const auto * hero = callback->getHero(subject);
			if(!hero || hero->getOwner() != playerID || hero->visitablePos() != expected)
				throw std::runtime_error("Contextual route changed before native issue");
			callback->moveHero(hero, hero->convertFromVisitablePos(step), routeStep.transit);
		}, true, decision, &issued);
		if(!issued || ended())
		{
			if(ended())
				return true;
			std::lock_guard lock(stateMutex);
			status = contextState.commands > 0 ? "interaction" : "interrupted";
			break;
		}
		{
			const auto gameLock = readGameState();
			captureContextual();
			const auto * hero = callback->getHero(subject);
			if(!hero || hero->getOwner() != playerID)
				status = "hero_unavailable";
			else
			{
				std::lock_guard lock(stateMutex);
				if(!accepted)
					status = "interrupted";
				else if(contextState.interactions != interactions)
					status = "interaction";
				else if(hero->visitablePos() != step)
					status = "interrupted";
				expected = hero->visitablePos();
			}
		}
		if(!status.empty())
			break;
	}
	JsonNode result;
	{
		const auto gameLock = readGameState();
		const auto * hero = callback->getHero(subject);
		if(!hero || hero->getOwner() != playerID)
			result = resultNode("hero_unavailable", true);
		else
		{
			const bool arrived = hero->visitablePos() == destination;
			if(status.empty())
				status = arrived ? "reached" : "movement_limit";
			result = resultNode(status.c_str(), true);
			result["position"] = coordinates(hero->visitablePos());
			result["movement"].Integer() = hero->movementPointsRemaining();
			result["destination_reached"].Bool() = arrived;
		}
	}
	contextualResult(std::move(result), decision);
	return true;
}

void ExplorerRL::showStructuredInfoDialog(EInfoWindowMode mode, const MetaString & text, const std::vector<Component> & components, int sound)
{
	if(!contextual)
	{
		CAdventureAI::showStructuredInfoDialog(mode, text, components, sound);
		return;
	}
	try
	{
		std::set<std::string> names;
		for(const auto * hero : callback->getHeroesInfo())
			if(hero->getOwner() == playerID)
				names.insert(hero->getNameTextID());
		JsonNode event;
		event["kind"].String() = "notice";
		event["notice"] = explorer_rl::normalizeNotice(mode, text, components, playerID, names, adventureDialogs, expandedInterface);
		contextualEvent(std::move(event));
	}
	catch(const explorer_rl::UnsupportedNotice &)
	{
		unsupported("unsupported_interaction: informational notice");
	}
}

void ExplorerRL::showInfoDialog(EInfoWindowMode mode, const std::string & text, const std::vector<Component> & components, int sound)
{
	if(contextual)
		unsupported("unsupported_interaction: unstructured informational notice");
	else
		CAdventureAI::showInfoDialog(mode, text, components, sound);
}

// These inherited notifications normally default to no-ops for an AI. The
// contextual contract requires a visible unsupported stop, including windows
// without a query identifier. Preserve the legacy interface's behavior.
void ExplorerRL::showShipyardDialog(const IShipyard *) { if(contextual) unsupported("unsupported_interaction: shipyard"); }
void ExplorerRL::showPuzzleMap() { if(contextual) unsupported("unsupported_interaction: puzzle map"); }
void ExplorerRL::viewWorldMap() { if(contextual) unsupported("unsupported_interaction: world map"); }
void ExplorerRL::showHillFortWindow(const CGObjectInstance *, const CGHeroInstance *) { if(contextual) unsupported("unsupported_interaction: hill fort"); }
void ExplorerRL::showThievesGuildWindow(const CGObjectInstance *) { if(contextual) unsupported("unsupported_interaction: thieves guild"); }
void ExplorerRL::showQuestLog() { if(contextual) unsupported("unsupported_interaction: quest log"); }
void ExplorerRL::askToAssembleArtifact(const ArtifactLocation & location)
{
	if(!contextual || expandedInterface)
		return;
	// PutArtifact/BulkMoveArtifacts also deliver this hint for ordinary trophies
	// that cannot be assembled. Match ArtifactsUIController::askToAssemble:
	// only a present artifact on an owned hero with an equipped combination
	// opens a prompt. No artifact details enter observations and no reply is sent.
	const auto * hero = callback->getHero(location.artHolder);
	if(!hero || hero->getOwner() != playerID)
		return;
	const auto * artifact = hero->getArt(location.slot);
	if(!artifact)
		return;
	if(!ArtifactUtils::assemblyPossibilities(hero, artifact->getTypeId(), true).empty())
		unsupported("unsupported_interaction: artifact assembly");
}
