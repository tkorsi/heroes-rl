#include "StdInc.h"
#include "ExplorerRL.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGHeroInstance.h"

void ExplorerRL::heroExchangeStarted(ObjectInstanceID first, ObjectInstanceID second, QueryID query)
{
	if(!armyManagement)
	{
		unsupported("hero exchange");
		return;
	}
	try
	{
		// A network callback already owns the stable applied packet. Only its
		// recipient's two owned heroes are supported; no nearby-hero search.
		const auto * a = callback->getHero(first);
		const auto * b = callback->getHero(second);
		if(!a || !b || a == b || a->getOwner() != playerID || b->getOwner() != playerID || query == QueryID(-1))
			throw std::runtime_error("unsupported_interaction: non-owned hero exchange");
		std::lock_guard lock(stateMutex);
		if(contextState.ownHeroExchange || contextState.context != "map" || !contextState.executing
			|| !contextState.routeInteractions || !contextState.decision)
			throw std::runtime_error("unsupported_interaction: unexpected hero exchange lifecycle");
		contextState.ownHeroExchange = explorer_rl::ContextualState::OwnHeroExchange{
			first.getNum(), second.getNum(), query.getNum(), false};
		++contextState.interactions;
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = "army_exchange_requested";
		record["event"]["subject"].String() = contextState.handle(first.getNum());
		record["event"]["partner"].String() = contextState.handle(second.getNum());
		contextState.records.push_back(std::move(record));
		changed.notify_all();
	}
	catch(const std::exception & error)
	{
		unsupported(error.what());
	}
}

bool ExplorerRL::activeOwnHeroExchange(int first, int second) const
{
	if(!armyManagement || !contextState.ownHeroExchange || !contextState.ownHeroExchange->installed)
		return false;
	const auto & exchange = *contextState.ownHeroExchange;
	return (exchange.first == first && exchange.second == second)
		|| (exchange.first == second && exchange.second == first);
}

void ExplorerRL::installOwnHeroExchange()
{
	if(!armyManagement || !contextState.ownHeroExchange || contextState.ownHeroExchange->installed)
		return;
	if(contextState.decision || contextState.executing || contextState.context != "map")
		throw std::runtime_error("Hero exchange cannot replace an unfinished parent action");
	auto & exchange = *contextState.ownHeroExchange;
	for(const int id : {exchange.first, exchange.second})
	{
		const auto * hero = callback->getHero(ObjectInstanceID(id));
		if(exchange.joiningNeutral && id == exchange.second)
		{
			const auto * source = callback->getObj(ObjectInstanceID(id), false);
			if(!source || source->ID != Obj::MONSTER || source->getOwner().isValidPlayer())
				throw std::runtime_error("Joining neutral became unavailable");
		}
		else if(!hero || hero->getOwner() != playerID)
			throw std::runtime_error("Owned hero exchange participant became unavailable");
	}
	exchange.installed = true;
	contextState.context = "exchange";
	contextState.subject = contextState.handle(exchange.first);
	contextState.selection = contextState.handle(exchange.second);
	contextState.panel.clear();
	auto record = contextState.record("context_opened", callback->getDate(Date::DAY));
	record["context"] = contextState.contextJson();
	record["revision"].Integer() = ++contextState.revision;
	record["reason"].String() = exchange.joiningNeutral ? "neutral_join" : "own_hero_exchange";
	contextState.records.push_back(std::move(record));
}

bool ExplorerRL::executeExchangeClose(const JsonNode & action, int64_t decision)
{
	for(const char * key : {"subject", "x", "y", "panel", "selection", "count", "destination"})
		if(!action[key].isNull()) throw std::runtime_error("Exchange close has no parameters");
	int query = -1;
	int64_t revision = -1;
	{
		const auto gameLock = readGameState();
		captureContextual();
		std::lock_guard lock(stateMutex);
		if(contextState.decision != decision || terminal || stopping) return false;
		if(!contextState.ownHeroExchange || !contextState.ownHeroExchange->installed || contextState.context != "exchange")
			throw std::runtime_error("No current native own-hero exchange to close");
		query = contextState.ownHeroExchange->query;
		revision = contextState.revision;
		contextState.executing = true;
		contextState.commands = 0;
		contextState.issueArmed = true;
		contextState.routeInteractions.reset();
		auto record = contextState.record("consumed", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["consumer"].String() = "engine";
		contextState.records.push_back(std::move(record));
	}
	bool issued = false;
	const bool accepted = executeRequest([&] { callback->selectionMade(0, QueryID(query)); }, true, decision, &issued, [&]
	{
		if(contextState.revision != revision || contextState.context != "exchange"
			|| !contextState.ownHeroExchange || contextState.ownHeroExchange->query != query)
			throw std::runtime_error("Native exchange changed before close");
		// Closing hides both army panels before the query reply resumes any
		// parent movement callbacks. The query itself is answered explicitly.
		contextState.context = "map";
		contextState.subject.clear();
		contextState.panel.clear();
		contextState.selection.clear();
		contextState.ownHeroExchange.reset();
		auto record = contextState.record("context", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["context"] = contextState.contextJson();
		record["revision"].Integer() = ++contextState.revision;
		contextState.records.push_back(std::move(record));
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	if(!accepted) throw std::runtime_error("Engine rejected native exchange closing");
	JsonNode result;
	result["status"].String() = "closed";
	result["accepted"].Bool() = true;
	for(const char * key : {"position", "movement", "destination_reached"}) result[key] = JsonNode();
	contextualResult(std::move(result), decision);
	return true;
}
