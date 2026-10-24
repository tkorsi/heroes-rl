#include "StdInc.h"
#include "ExplorerRL.h"
#include "KnowledgePlanner.h"
#include "VisibleWorld.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/callback/ExperimentHooks.h"
#include "../../lib/gameState/CGameState.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"
#include "../../lib/networkPacks/Component.h"
#include "../../lib/networkPacks/EInfoWindowMode.h"
#include "../../lib/StartInfo.h"
#include "../../lib/texts/MetaString.h"

#include <limits>

/// adventure-windows-v1: the contextual interface selected with
/// EXPLORER_RL_INTERFACE. Each decision offers the current context's actions;
/// the policy answers with one action bound to that decision and revision.
/// Opening and closing windows is local navigation consumed here without a
/// game command. A destination is routed over the player's own knowledge and
/// walked one native step per command, each acknowledged by the server, until
/// the current-turn prefix ends or the first interaction, rejection or loss
/// stops it. Every event the player may see is reported in order, including
/// the consumption, issue and acknowledgement of each command, and anything
/// this interface cannot represent stops the run as unsupported.
namespace
{
JsonNode positionJson(int x, int y, int z)
{
	JsonNode result;
	for(const int coordinate : {x, y, z})
		result.Vector().emplace_back(coordinate);
	return result;
}

JsonNode positionJson(const std::array<int, 3> & position)
{
	return positionJson(position[0], position[1], position[2]);
}

JsonNode tileJson(const std::array<int, 3> & position, const explorer_rl::ContextTile & tile)
{
	JsonNode result;
	result["x"].Integer() = position[0];
	result["y"].Integer() = position[1];
	result["z"].Integer() = position[2];
	result["terrain"].Integer() = tile.terrain;
	result["road"].Integer() = tile.road;
	result["blocked"].Bool() = tile.blocked;
	result["visitable"].Bool() = tile.visitable;
	return result;
}

JsonNode objectJson(const explorer_rl::VisibleObject & object)
{
	JsonNode result;
	result["id"].Integer() = object.id; // routing only; the adapter replaces it with an opaque handle
	result["x"].Integer() = object.x;
	result["y"].Integer() = object.y;
	result["z"].Integer() = object.z;
	result["type"].Integer() = object.type;
	result["subtype"].Integer() = object.subtype;
	result["owner"].Integer() = object.owner;
	return result;
}

JsonNode ownJson(const explorer_rl::ContextOwn & own)
{
	JsonNode result;
	auto & resources = result["resources"].Vector();
	for(const auto amount : own.resources)
		resources.emplace_back(amount);
	auto & heroes = result["heroes"].Vector();
	heroes.clear();
	for(const auto & hero : own.heroes)
	{
		JsonNode entry;
		entry["id"].Integer() = hero.id;
		entry["position"] = positionJson(hero.position);
		entry["movement"].Integer() = hero.movement;
		entry["movement_max"].Integer() = hero.movementMax;
		heroes.push_back(std::move(entry));
	}
	auto & towns = result["towns"].Vector();
	towns.clear();
	for(const auto & town : own.towns)
	{
		JsonNode entry;
		entry["id"].Integer() = town.id;
		entry["position"] = positionJson(town.position);
		towns.push_back(std::move(entry));
	}
	return result;
}

JsonNode changeJson(const explorer_rl::ContextChange & change, int64_t sequence, int day)
{
	JsonNode event;
	event["kind"].String() = "context_update";
	event["sequence"].Integer() = sequence;
	event["day"].Integer() = day;
	for(const auto * key : {"appeared", "changed", "disappeared", "revealed", "tiles_changed", "hidden"})
		event[key].Vector().clear();
	for(const auto & object : change.appeared)
		event["appeared"].Vector().push_back(objectJson(object));
	for(const auto & object : change.changed)
		event["changed"].Vector().push_back(objectJson(object));
	for(const auto id : change.disappeared)
		event["disappeared"].Vector().emplace_back(id);
	for(const auto & fact : change.revealed)
		event["revealed"].Vector().push_back(tileJson(fact.position, fact.tile));
	for(const auto & fact : change.tilesChanged)
		event["tiles_changed"].Vector().push_back(tileJson(fact.position, fact.tile));
	for(const auto & tile : change.hidden)
		event["hidden"].Vector().push_back(positionJson(tile));
	event["own"] = change.own ? ownJson(*change.own) : JsonNode();
	return event;
}

JsonNode marker(const char * kind)
{
	JsonNode event;
	event["kind"].String() = kind;
	return event;
}

JsonNode consumed(const char * consumer)
{
	auto event = marker("consumed");
	event["consumer"].String() = consumer;
	return event;
}

int64_t integerField(const JsonNode & node, const std::string & key)
{
	if(node.getType() != JsonNode::JsonType::DATA_STRUCT || !node.Struct().count(key)
		|| node.Struct().at(key).getType() != JsonNode::JsonType::DATA_INTEGER)
		throw std::runtime_error("Policy response requires integer " + key);
	return node.Struct().at(key).Integer();
}

explorer_rl::NoticeDisplay noticeDisplay(EInfoWindowMode mode)
{
	switch(mode)
	{
		case EInfoWindowMode::AUTO:
			return explorer_rl::NoticeDisplay::AUTO;
		case EInfoWindowMode::MODAL:
			return explorer_rl::NoticeDisplay::MODAL;
		case EInfoWindowMode::INFO:
			return explorer_rl::NoticeDisplay::INFO;
	}
	throw std::runtime_error("Unknown information window mode");
}

JsonNode noticeJson(const explorer_rl::Notice & notice)
{
	JsonNode result;
	result["display_mode"].String() = notice.displayMode;
	auto & text = result["text"].Vector();
	text.clear();
	for(const auto & token : notice.text)
	{
		JsonNode entry;
		entry["op"].String() = token.op;
		entry["text_id"] = token.textID ? JsonNode(*token.textID) : JsonNode();
		entry["number"] = token.number ? JsonNode(*token.number) : JsonNode();
		entry["owner"] = token.owner ? JsonNode(*token.owner) : JsonNode();
		text.push_back(std::move(entry));
	}
	auto & components = result["components"].Vector();
	components.clear();
	for(const auto & component : notice.components)
	{
		JsonNode entry;
		entry["kind"].String() = component.kind;
		entry["resource"] = component.resource ? JsonNode(*component.resource) : JsonNode();
		entry["amount"] = component.amount ? JsonNode(static_cast<int64_t>(*component.amount)) : JsonNode();
		entry["owner"] = component.owner ? JsonNode(*component.owner) : JsonNode();
		components.push_back(std::move(entry));
	}
	return result;
}
}

JsonNode ExplorerRL::contextualReady()
{
	auto ready = message("ready");
	ready["interface"].String() = "adventure-windows-v1";
	ready["action_schema"].String() = "heroes-context-actions-v2";
	ready["observation_events"].String() = "context_update-v1";
	ready["routing"].String() = "player-knowledge-current-turn-v1";
	ready["notices"].String() = "audited-notice-v1";
	ready["battle_ai"].String() = getBattleAIName();
	ready["profiling"].Bool() = profile;
	ready["snapshot_every"].Integer() = snapshotEvery;
	const auto gameLock = readGameState();
	const auto size = callback->getMapSize();
	ready["map_size"] = positionJson(size.x, size.y, size.z); // public map dimensions
	return ready;
}

void ExplorerRL::pushEvent(JsonNode event, bool stopCause)
{
	std::lock_guard lock(stateMutex);
	events.push_back(std::move(event));
	if(stopCause)
		++stopCauses;
}

void ExplorerRL::noteContextChanges()
{
	// Callers guarantee a stable state, as for noteVisibleChanges().
	const int day = callback->getDate(Date::DAY);
	if(day < 1)
		return;
	const auto start = std::chrono::steady_clock::now();
	auto snapshot = explorer_rl::captureContext(explorer_rl::CallbackContextWorld{*callback, playerID, callback->getMapSize()});
	const auto position = experimentPackPosition();
	const uint64_t digest = ledgerEnabled ? explorer_rl::contextDigest(snapshot) : 0;
	std::lock_guard lock(stateMutex);
	const auto change = contextTracker.update(std::move(snapshot));
	++visibleCaptures;
	visibleCaptureMicros += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
	lastCapture = Capture{position.index, position.type, digest};
	if(change.empty())
		return;
	if(observationLedger)
		std::fprintf(observationLedger, "{\"kind\":\"context_capture\",\"sequence\":%lld,\"pack\":%llu,\"pack_type\":%u,\"digest\":\"%016llx\"}\n",
			static_cast<long long>(visibleSequence), static_cast<unsigned long long>(position.index),
			static_cast<unsigned>(position.type), static_cast<unsigned long long>(digest));
	events.push_back(changeJson(change, visibleSequence++, day));
	for(const auto id : change.heroesLost)
	{
		auto lost = marker("hero_lost");
		lost["hero_id"].Integer() = id;
		events.push_back(std::move(lost));
	}
}

JsonNode ExplorerRL::takeContextEvents()
{
	std::lock_guard lock(stateMutex);
	JsonNode result;
	auto & taken = result.Vector();
	taken.clear();
	for(auto & event : events)
	{
		if(event["kind"].String() != "_ack")
		{
			taken.push_back(std::move(event));
			continue;
		}
		// Acknowledgements of other requests (turn queries, battle actions) are not commands of this action.
		const auto command = commandRequests.find(static_cast<int>(event["request"].Integer()));
		if(command == commandRequests.end())
			continue;
		const bool accepted = event["accepted"].Bool();
		auto acknowledged = marker("acknowledged");
		acknowledged["command_index"].Integer() = command->second;
		acknowledged["accepted"].Bool() = accepted;
		taken.push_back(std::move(acknowledged));
		// The turn ends where the server accepted it; what follows belongs to the next turn.
		if(action && action->endTurn && accepted)
			taken.push_back(marker("turn_end"));
	}
	events.clear();
	// The action's result leaves with these events: it is no longer gameOver's to report.
	action.reset();
	commandRequests.clear();
	return result;
}

JsonNode ExplorerRL::contextSnapshotJson(const explorer_rl::ContextSnapshot & snapshot) const
{
	JsonNode result;
	auto & tiles = result["tiles"].Vector();
	tiles.clear();
	for(size_t index = 0; index < snapshot.tiles.size(); ++index)
	{
		if(!snapshot.tiles[index])
			continue;
		const int x = static_cast<int>(index % snapshot.width);
		const int y = static_cast<int>((index / snapshot.width) % snapshot.height);
		const int z = static_cast<int>(index / (static_cast<size_t>(snapshot.width) * snapshot.height));
		tiles.push_back(tileJson({x, y, z}, snapshot.facts[index]));
	}
	auto & objects = result["objects"].Vector();
	objects.clear();
	for(const auto & [id, object] : snapshot.objects)
		objects.push_back(objectJson(object));
	result["own"] = ownJson(snapshot.own);
	return result;
}

JsonNode ExplorerRL::heroWindow(const CGHeroInstance & hero) const
{
	JsonNode result;
	result["kind"].String() = "hero";
	result["subject"].Integer() = hero.id.getNum();
	auto & army = result["army"].Vector();
	army.clear();
	for(const auto & [slot, stack] : hero.Slots())
	{
		JsonNode entry;
		entry["slot"].Integer() = slot.getNum();
		entry["creature"].Integer() = stack->getCreatureID().getNum();
		entry["count"].Integer() = stack->getCount();
		army.push_back(std::move(entry));
	}
	auto & skills = result["primary_skills"].Vector();
	for(const auto skill : {PrimarySkill::ATTACK, PrimarySkill::DEFENSE, PrimarySkill::SPELL_POWER, PrimarySkill::KNOWLEDGE})
		skills.emplace_back(hero.getPrimSkillLevel(skill));
	result["mana"].Integer() = hero.mana;
	result["level"].Integer() = hero.level;
	return result;
}

JsonNode ExplorerRL::townWindow(const CGTownInstance & town) const
{
	JsonNode result;
	result["kind"].String() = "town";
	result["subject"].Integer() = town.id.getNum();
	auto & army = result["army"].Vector();
	army.clear();
	for(const auto & [slot, stack] : town.Slots())
	{
		JsonNode entry;
		entry["slot"].Integer() = slot.getNum();
		entry["creature"].Integer() = stack->getCreatureID().getNum();
		entry["count"].Integer() = stack->getCount();
		army.push_back(std::move(entry));
	}
	result["faction"].Integer() = town.getFactionID().getNum();
	return result;
}

JsonNode ExplorerRL::contextResult(const ContextAction & current)
{
	// Callers hold the stable state and stateMutex.
	JsonNode result;
	if(current.endTurn)
	{
		const bool ended = current.accepted == 1;
		result["status"].String() = ended ? "turn_ended" : "rejected";
		result["accepted"].Bool() = ended;
		return result;
	}
	const bool accepted = current.accepted > 0;
	const auto * hero = dynamic_cast<const CGHeroInstance *>(callback->getObj(current.hero, false));
	if(!hero || hero->getOwner() != playerID)
	{
		result["status"].String() = "hero_unavailable";
		result["accepted"].Bool() = accepted;
		return result;
	}
	const auto position = hero->visitablePos();
	const bool reached = position == current.destination;
	// A visit, battle, notice or loss after the last accepted step stopped the route there.
	const bool interaction = accepted && stopCauses != current.causesAtIssue;
	const char * status = current.rejected ? (accepted ? "interrupted" : "rejected")
		: interaction ? "interaction"
		: current.deviated || !current.finished ? "interrupted"
		: reached ? "reached" : "movement_limit";
	result["status"].String() = status;
	result["accepted"].Bool() = accepted;
	result["position"] = positionJson(position.x, position.y, position.z);
	result["movement"].Integer() = hero->movementPointsRemaining();
	result["destination_reached"].Bool() = reached;
	return result;
}

void ExplorerRL::settleCommand(int command, bool accepted)
{
	// Requires stateMutex. Called once both the command's request ID and its
	// acknowledgement are known, whichever arrived last.
	if(!action || !action->pending || command != action->commands - 1)
		return;
	action->pending = false;
	action->lastAccepted = accepted;
	if(accepted)
		++action->accepted;
	else
		action->rejected = true;
}

ExplorerRL::Command ExplorerRL::issueCommand(const std::function<void()> & request, const std::function<bool()> & precondition)
{
	{
		std::lock_guard lock(stateMutex);
		pendingRequest = -1;
		acknowledgements.clear();
	}
	{
		const auto gameLock = readGameState();
		// As in executeRequest: a command the game ended before is never issued.
		const bool playing = callback->getPlayerStatus(playerID, false) == EPlayerStatus::INGAME;
		const bool ready = playing && (!precondition || precondition());
		{
			std::lock_guard lock(stateMutex);
			if(!failure.empty())
				throw std::runtime_error(failure);
			if(!playing || stopping || terminal)
				return Command::ENDED;
			if(!action)
				throw std::logic_error("A native command needs an in-flight action");
			if(!ready)
			{
				action->deviated = true;
				return Command::DECLINED;
			}
			issuingCommand = action->commands++;
			action->pending = true;
			action->causesAtIssue = stopCauses;
			auto issued = marker("issued");
			issued["command_index"].Integer() = issuingCommand;
			events.push_back(std::move(issued));
		}
		request();
	}
	std::unique_lock lock(stateMutex);
	issuingCommand = -1;
	const int requestID = pendingRequest;
	if(requestID < 0)
		throw std::runtime_error("Engine did not send the requested command");
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
	while(true)
	{
		if(!failure.empty())
			throw std::runtime_error(failure);
		if(stopping || terminal)
			return Command::ENDED;
		const auto acknowledged = acknowledgements.find(requestID);
		if(acknowledged != acknowledgements.end() && !moving && !inBattle)
		{
			acknowledgements.erase(acknowledged);
			if(!action || action->pending)
				throw std::logic_error("A command acknowledgement was not settled");
			return action->lastAccepted ? Command::ACCEPTED : Command::REJECTED;
		}
		if(changed.wait_until(lock, deadline) == std::cv_status::timeout)
			throw std::runtime_error("Engine command or battle completion timed out");
	}
}

std::optional<JsonNode> ExplorerRL::planMove(const CGHeroInstance & hero, int64_t decision, const int3 & destination, std::vector<int3> & steps)
{
	// Callers hold the stable state and validated that the player owns the hero.
	const auto origin = hero.visitablePos();
	const char * immediate = nullptr;
	steps.clear();
	if(origin == destination)
		immediate = "reached";
	else if(hero.movementPointsRemaining() <= 0)
		immediate = "movement_limit";
	else
	{
		auto plan = explorer_rl::planRoute(*callback, hero, destination);
		if(!plan.known)
			immediate = "no_route_known";
		else if(plan.steps.empty())
			immediate = "movement_limit";
		else
			steps = std::move(plan.steps);
	}
	if(immediate)
	{
		// No native command: the adapter consumed the decision and nothing changed.
		JsonNode result;
		result["status"].String() = immediate;
		result["accepted"].Bool() = true;
		result["position"] = positionJson(origin.x, origin.y, origin.z);
		result["movement"].Integer() = hero.movementPointsRemaining();
		result["destination_reached"].Bool() = origin == destination;
		pushEvent(consumed("adapter"));
		return result;
	}
	std::lock_guard lock(stateMutex);
	action = ContextAction{};
	action->decision = decision;
	action->hero = hero.id;
	action->destination = destination;
	events.push_back(consumed("engine"));
	return std::nullopt;
}

void ExplorerRL::runContextualTurn(QueryID queryID)
{
	if(queryID != QueryID(-1) && !executeRequest([this, queryID] { callback->selectionMade(0, queryID); }))
	{
		if(!ended())
			throw std::runtime_error("Engine rejected turn acknowledgement");
		return;
	}
	while(!ended())
	{
		const auto id = decisionID++;
		const auto decisionRevision = ++revision;
		const bool sent = buildAndSend([&]
		{
			const auto buildStart = std::chrono::steady_clock::now();
			auto request = message("decision");
			request["decision_id"].Integer() = id;
			request["revision"].Integer() = decisionRevision;
			JsonNode context;
			context["kind"].String() = !window ? "map" : window->town ? "town" : "hero";
			context["subject"] = window ? JsonNode(static_cast<int64_t>(window->subject.getNum())) : JsonNode();
			request["context"] = context;
			{
				// No packet can be applied while this lock is held, so the events end
				// exactly at the state this decision is offered in.
				const auto gameLock = readGameState();
				noteContextChanges();
				request["day"].Integer() = callback->getDate(Date::DAY);
				request["events"] = takeContextEvents();
				if(snapshotEvery > 0 && observations++ % snapshotEvery == 0)
				{
					// The capture repeats what the events established, so the adapter can verify its fold.
					std::lock_guard lock(stateMutex);
					if(contextTracker.latest())
						request["state"] = contextSnapshotJson(*contextTracker.latest());
				}
				noteLedgerMessage("decision");
			}
			if(profile)
			{
				request["perf"] = takeProfile();
				request["perf"]["build_us"].Integer() = std::chrono::duration_cast<std::chrono::microseconds>(
					std::chrono::steady_clock::now() - buildStart).count();
			}
			return request;
		});
		if(!sent)
			return;
		const auto response = receiveMessage();
		if(ended())
			return;
		// Every response is bound to the decision it answers and to the context revision it was offered in.
		if(integerField(response, "decision_id") != id || integerField(response, "revision") != decisionRevision)
			throw std::runtime_error("Policy response does not answer the current decision and revision");
		if(response.Struct().size() != 3 || !response.Struct().count("action")
			|| response.Struct().at("action").getType() != JsonNode::JsonType::DATA_STRUCT
			|| response.Struct().at("action").Struct().count("kind") == 0
			|| response.Struct().at("action").Struct().at("kind").getType() != JsonNode::JsonType::DATA_STRING)
			throw std::runtime_error("Policy response requires exactly one typed contextual action");
		const auto & chosen = response.Struct().at("action");
		const auto kind = chosen.Struct().at("kind").String();

		/// The result is computed under the same stable state its events end at.
		const auto sendResult = [&](const std::function<JsonNode()> & outcome, std::optional<JsonNode> opened = std::nullopt)
		{
			return buildAndSend([&]
			{
				auto reply = message("result");
				reply["decision_id"].Integer() = id;
				{
					const auto gameLock = readGameState();
					noteContextChanges();
					reply["result"] = outcome();
					reply["events"] = takeContextEvents();
					noteLedgerMessage("result");
				}
				if(opened)
					reply["window"] = std::move(*opened);
				if(profile)
					reply["perf"] = takeProfile();
				return reply;
			});
		};
		const auto fixed = [](JsonNode outcome) { return [outcome] { return outcome; }; };
		const auto actionResult = [this]
		{
			std::lock_guard lock(stateMutex);
			if(!action)
				throw std::logic_error("A command result needs its in-flight action");
			return contextResult(*action);
		};
		const auto navigate = [&](const char * contextKind, std::optional<int64_t> subject)
		{
			// Local navigation: consumed here, with an explicit context change and no game command.
			pushEvent(consumed("adapter"));
			auto change = marker("context");
			change["context"]["kind"].String() = contextKind;
			change["context"]["subject"] = subject ? JsonNode(*subject) : JsonNode();
			change["revision"].Integer() = ++revision;
			pushEvent(std::move(change));
		};

		if(window)
		{
			// A window offers only close; nothing in the game changed while it was open.
			if(kind != "close" || chosen.Struct().size() != 1)
				throw std::runtime_error("Policy response is not offered in a window");
			navigate("map", std::nullopt);
			window.reset();
			JsonNode outcome;
			outcome["status"].String() = "closed";
			outcome["accepted"].Bool() = true;
			if(!sendResult(fixed(std::move(outcome))))
				return;
			continue;
		}

		if(kind == "end_turn")
		{
			if(chosen.Struct().size() != 1)
				throw std::runtime_error("End turn takes no parameters");
			{
				std::lock_guard lock(stateMutex);
				action = ContextAction{};
				action->decision = id;
				action->endTurn = true;
				events.push_back(consumed("engine"));
			}
			const auto outcome = issueCommand([this] { callback->endTurn(); });
			if(outcome == Command::ENDED)
				return; // gameOver reports the result if the acknowledgement arrived
			if(!sendResult(actionResult))
				return;
			if(outcome == Command::ACCEPTED)
				return; // the next decision comes with the player's next turn
			continue;
		}

		if(kind == "open_hero" || kind == "open_town")
		{
			if(chosen.Struct().size() != 2)
				throw std::runtime_error("Opening a window takes exactly one subject");
			const ObjectInstanceID subject(static_cast<si32>(integerField(chosen, "subject")));
			JsonNode opened;
			{
				const auto gameLock = readGameState();
				const auto * object = callback->getObj(subject, false);
				const auto * hero = dynamic_cast<const CGHeroInstance *>(object);
				const auto * town = dynamic_cast<const CGTownInstance *>(object);
				if(kind == "open_hero" ? !hero || hero->getOwner() != playerID : !town || town->getOwner() != playerID)
					throw std::runtime_error("Policy opened a window of an object the player does not own");
				opened = hero ? heroWindow(*hero) : townWindow(*town);
			}
			const bool town = kind == "open_town";
			navigate(town ? "town" : "hero", subject.getNum());
			window = OpenWindow{town, subject};
			JsonNode outcome;
			outcome["status"].String() = "opened";
			outcome["accepted"].Bool() = true;
			if(!sendResult(fixed(std::move(outcome)), std::move(opened)))
				return;
			continue;
		}

		if(kind != "move_to" || chosen.Struct().size() != 4)
			throw std::runtime_error("Policy response is not a supported contextual action");
		const ObjectInstanceID heroID(static_cast<si32>(integerField(chosen, "subject")));
		const auto x = integerField(chosen, "x");
		const auto y = integerField(chosen, "y");
		std::vector<int3> steps;
		int3 expected;
		std::optional<JsonNode> immediate;
		{
			const auto gameLock = readGameState();
			const auto * hero = dynamic_cast<const CGHeroInstance *>(callback->getObj(heroID, false));
			if(!hero || hero->getOwner() != playerID)
				throw std::runtime_error("Policy moved a hero the player does not own");
			const auto size = callback->getMapSize();
			if(x < 0 || y < 0 || x >= size.x || y >= size.y)
				throw std::runtime_error("Policy destination is outside the public map bounds");
			expected = hero->visitablePos();
			immediate = planMove(*hero, id, int3(static_cast<int>(x), static_cast<int>(y), expected.z), steps);
		}
		if(immediate)
		{
			if(!sendResult(fixed(std::move(*immediate))))
				return;
			continue;
		}
		// One adjacent native step per command, each from where the previous one
		// left the hero. A rejection, deviation or interaction ends the walk.
		bool walked = true;
		for(const auto & step : steps)
		{
			const auto outcome = issueCommand([this, heroID, step]
			{
				const auto * hero = callback->getHero(heroID);
				callback->moveHero(hero, hero->convertFromVisitablePos(step), false);
			}, [this, heroID, expected]
			{
				const auto * hero = dynamic_cast<const CGHeroInstance *>(callback->getObj(heroID, false));
				return hero && hero->getOwner() == playerID && hero->visitablePos() == expected;
			});
			if(outcome == Command::ENDED)
				return; // gameOver reports the result once every issued step is acknowledged
			bool stop = outcome != Command::ACCEPTED;
			if(!stop)
			{
				std::lock_guard lock(stateMutex);
				stop = stopCauses != action->causesAtIssue;
			}
			if(stop)
			{
				walked = false;
				break;
			}
			expected = step;
		}
		{
			std::lock_guard lock(stateMutex);
			action->finished = walked;
		}
		if(!sendResult(actionResult))
			return;
	}
}

void ExplorerRL::showInfoNotice(EInfoWindowMode mode, const MetaString & text, const std::vector<Component> & components)
{
	if(!contextual)
		return;
	// The packet thread delivers this immediately before showInfoDialog on the same interface.
	if(pendingNotice)
	{
		unsupported("structured notice without its information window");
		return;
	}
	const auto operations = text.operations();
	if(!operations)
	{
		pendingNotice = explorer_rl::NoticeParse{std::nullopt, "message with incomplete or unknown parts"};
		return;
	}
	static_assert(static_cast<int>(MetaStringOperation::Kind::APPEND_EOL) == static_cast<int>(explorer_rl::NoticeOperationKind::APPEND_EOL));
	static_assert(static_cast<int>(MetaStringOperation::Kind::REPLACE_POSITIVE_NUMBER)
		== static_cast<int>(explorer_rl::NoticeOperationKind::REPLACE_POSITIVE_NUMBER));
	std::vector<explorer_rl::NoticeOperation> inputs;
	for(const auto & operation : *operations)
		inputs.push_back({static_cast<explorer_rl::NoticeOperationKind>(operation.kind), operation.text, operation.number});
	std::vector<explorer_rl::NoticeComponentInput> componentInputs;
	for(const auto & component : components)
	{
		using Input = explorer_rl::NoticeComponentInput;
		Input input;
		input.kind = component.type == ComponentType::RESOURCE ? Input::Kind::RESOURCE
			: component.type == ComponentType::RESOURCE_PER_DAY ? Input::Kind::RESOURCE_PER_DAY
			: component.type == ComponentType::FLAG ? Input::Kind::FLAG : Input::Kind::OTHER;
		// A variant equals its own alternative rebuilt from as<>() only when it holds that alternative.
		const auto resource = component.subType.as<GameResID>();
		const auto player = component.subType.as<PlayerColor>();
		if(component.subType == ComponentSubType(resource))
		{
			input.subtype = Input::Subtype::RESOURCE;
			input.number = resource.getNum();
		}
		else if(component.subType == ComponentSubType(player))
		{
			input.subtype = Input::Subtype::PLAYER;
			input.number = player.getNum();
		}
		input.value = component.value;
		componentInputs.push_back(input);
	}
	// The scenario's only other player; with more, any player reference is unsupported.
	int opponent = std::numeric_limits<int>::min();
	int others = 0;
	for(const auto & [color, info] : callback->getStartInfo()->playerInfos)
		if(color != playerID && ++others == 1)
			opponent = color.getNum();
	if(others != 1)
		opponent = std::numeric_limits<int>::min();
	pendingNotice = explorer_rl::parseNotice(noticeDisplay(mode), inputs, componentInputs, playerID.getNum(), opponent);
}

void ExplorerRL::showInfoDialog(EInfoWindowMode, const std::string &, const std::vector<Component> &, int)
{
	if(!contextual)
		return;
	auto parse = std::exchange(pendingNotice, std::nullopt);
	if(!parse)
	{
		unsupported("information window the client composed");
		return;
	}
	if(!parse->notice)
	{
		unsupported("notice: " + parse->unsupported);
		return;
	}
	auto event = marker("notice");
	event["notice"] = noticeJson(*parse->notice);
	pushEvent(std::move(event), true);
}

// Special windows have no contextual representation yet: each stops the run.
void ExplorerRL::showShipyardDialog(const IShipyard *) { if(contextual) unsupported("shipyard window"); }
void ExplorerRL::showPuzzleMap() { if(contextual) unsupported("puzzle map"); }
void ExplorerRL::viewWorldMap() { if(contextual) unsupported("world map"); }
void ExplorerRL::showHillFortWindow(const CGObjectInstance *, const CGHeroInstance *) { if(contextual) unsupported("hill fort"); }
void ExplorerRL::showThievesGuildWindow(const CGObjectInstance *) { if(contextual) unsupported("thieves guild"); }
void ExplorerRL::showQuestLog() { if(contextual) unsupported("quest log"); }
void ExplorerRL::showWorldViewEx(const std::vector<ObjectPosInfo> &, bool) { if(contextual) unsupported("world view"); }
