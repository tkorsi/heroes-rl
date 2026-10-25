#include "StdInc.h"
#include "ExplorerRL.h"
#include "Dialog.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGHeroInstance.h"

void ExplorerRL::showStructuredBlockingDialog(const MetaString & text, const std::vector<Component> & components,
	QueryID query, int sound, bool selection, bool cancel, bool safeToAutoaccept)
{
	if(!adventureDialogs)
	{
		CAdventureAI::showStructuredBlockingDialog(text, components, query, sound, selection, cancel, safeToAutoaccept);
		return;
	}
	try
	{
		// Normalize the delivered window, never the object's hidden reward or
		// disposition. safeToAutoaccept never supplies a strategic answer.
		auto normalized = explorer_rl::normalizeDialog(text, components, selection, cancel);
		std::lock_guard lock(stateMutex);
		if(query == QueryID(-1) || contextState.adventureDialog || contextState.ownHeroExchange
			|| contextState.context != "map" || !contextState.executing
			|| !contextState.decision || !contextState.executingHero)
			throw std::runtime_error("unsupported_interaction: unexpected adventure dialog lifecycle");
		const int heroID = *contextState.executingHero;
		const auto * hero = callback->getHero(ObjectInstanceID(heroID));
		if(!hero || hero->getOwner() != playerID)
			throw std::runtime_error("unsupported_interaction: adventure dialog without its owned actor");
		const std::string ref = "dialog-" + std::to_string(contextState.nextDialog++);
		normalized.panel["ref"].String() = ref;
		auto & choices = normalized.panel["choices"].Vector();
		if(choices.empty() || choices.size() != normalized.replies.size())
			throw std::runtime_error("Unsupported adventure dialog choices");
		explorer_rl::ContextualState::AdventureDialog pending{heroID, query.getNum(), normalized.panel, {}};
		for(size_t index = 0; index < choices.size(); ++index)
		{
			const std::string choice = ref + "-choice-" + std::to_string(index);
			pending.panel["choices"].Vector().at(index)["ref"].String() = choice;
			pending.replies.emplace(choice, normalized.replies.at(index));
		}
		auto record = contextState.record("event", callback->getDate(Date::DAY));
		record["event"]["kind"].String() = "dialog_requested";
		record["event"]["subject"].String() = contextState.handle(heroID);
		record["event"]["dialog"] = pending.panel;
		contextState.records.push_back(std::move(record));
		contextState.adventureDialog = std::move(pending);
		++contextState.interactions;
		changed.notify_all();
	}
	catch(const std::exception &)
	{
		// No raw native text, query IDs or hidden payload in diagnostics.
		unsupported("unsupported_interaction: adventure dialog");
	}
}

void ExplorerRL::installAdventureDialog()
{
	if(!adventureDialogs || !contextState.adventureDialog || contextState.adventureDialog->installed)
		return;
	if(contextState.decision || contextState.executing || contextState.context != "map" || contextState.ownHeroExchange)
		throw std::runtime_error("Adventure dialog cannot replace an unfinished parent action");
	auto & dialog = *contextState.adventureDialog;
	const auto * hero = callback->getHero(ObjectInstanceID(dialog.hero));
	if(!hero || hero->getOwner() != playerID)
		throw std::runtime_error("Adventure dialog actor became unavailable");
	dialog.installed = true;
	contextState.context = "dialog";
	contextState.subject = contextState.handle(dialog.hero);
	contextState.selection = dialog.panel["ref"].String();
	contextState.panel.clear();
	auto record = contextState.record("context_opened", callback->getDate(Date::DAY));
	record["context"] = contextState.contextJson();
	record["revision"].Integer() = ++contextState.revision;
	record["reason"].String() = "adventure_dialog";
	contextState.records.push_back(std::move(record));
}

JsonNode ExplorerRL::adventureDialogPanel() const
{
	if(!adventureDialogs || !contextState.adventureDialog || !contextState.adventureDialog->installed
		|| contextState.context != "dialog" || contextState.selection != contextState.adventureDialog->panel["ref"].String())
		throw std::runtime_error("No explicitly installed adventure dialog");
	return contextState.adventureDialog->panel;
}

bool ExplorerRL::executeDialogAction(const JsonNode & action, int64_t decision)
{
	if(!adventureDialogs || action["kind"].String() != "choose_dialog"
		|| !action["subject"].isString() || !action["selection"].isString())
		throw std::runtime_error("Adventure dialog requires an explicit displayed choice");
	for(const char * key : {"x", "y", "panel", "count", "destination"})
		if(!action[key].isNull()) throw std::runtime_error("Unused adventure dialog action parameter");
	int query = -1, reply = -1;
	int64_t revision = -1;
	{
		const auto gameLock = readGameState();
		captureContextual();
		std::lock_guard lock(stateMutex);
		if(contextState.decision != decision || terminal || stopping) return false;
		if(!contextState.adventureDialog || !contextState.adventureDialog->installed
			|| contextState.context != "dialog" || contextState.subject != action["subject"].String())
			throw std::runtime_error("Choice does not address the current adventure dialog");
		const auto & dialog = *contextState.adventureDialog;
		const auto found = dialog.replies.find(action["selection"].String());
		if(found == dialog.replies.end()) throw std::runtime_error("Adventure choice was not displayed");
		query = dialog.query;
		reply = found->second;
		revision = contextState.revision;
		contextState.executing = true;
		contextState.executingHero = dialog.hero;
		contextState.commands = 0;
		contextState.issueArmed = true;
		contextState.routeInteractions.reset();
		auto record = contextState.record("consumed", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["consumer"].String() = "engine";
		contextState.records.push_back(std::move(record));
	}
	bool issued = false;
	const bool accepted = executeRequest([&] { callback->selectionMade(reply, QueryID(query)); }, true, decision, &issued, [&]
	{
		if(contextState.revision != revision || contextState.context != "dialog"
			|| !contextState.adventureDialog || contextState.adventureDialog->query != query)
			throw std::runtime_error("Adventure dialog changed before its native reply");
		// Clear the answered query before the reply can produce another dialog.
		// The new callback retains its own immutable payload and explicit choice.
		contextState.context = "map";
		contextState.subject.clear();
		contextState.panel.clear();
		contextState.selection.clear();
		contextState.adventureDialog.reset();
		auto record = contextState.record("context", callback->getDate(Date::DAY));
		record["decision_id"].Integer() = decision;
		record["context"] = contextState.contextJson();
		record["revision"].Integer() = ++contextState.revision;
		contextState.records.push_back(std::move(record));
		return true;
	});
	if(ended()) return true;
	if(!issued) return false;
	if(!accepted) throw std::runtime_error("Engine rejected the native adventure dialog reply");
	JsonNode result;
	result["status"].String() = "dialog_answered";
	result["accepted"].Bool() = true;
	for(const char * key : {"position", "movement", "destination_reached"}) result[key] = JsonNode();
	contextualResult(std::move(result), decision);
	return true;
}
