#include "StdInc.h"
#include "ContextualEvidence.h"

#include "../../lib/networkPacks/NetPacksBase.h"
#include "../../lib/callback/CGameInterface.h"
#include "../../lib/networkPacks/PacksForClient.h"
#include "../../lib/serializer/BinarySerializer.h"
#include "../../lib/serializer/CTypeList.h"

#include <chrono>
#include <cstdlib>

namespace explorer_rl
{
namespace
{
std::shared_ptr<ContextualEvidence> recorder;
thread_local std::string transactionActor;

std::string line(const JsonNode & value)
{
	auto result = value.toCompactString();
	result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
	result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
	return result;
}

JsonNode position()
{
	const auto value = experimentReceiptPosition();
	JsonNode result;
	result["index"].Integer() = value.index;
	result["type"].Integer() = value.type;
	result["completed"].Integer() = value.completed;
	switch(value.phase)
	{
	case ExperimentPacketPhase::Before: result["phase"].String() = "before"; break;
	case ExperimentPacketPhase::Applied: result["phase"].String() = "applied"; break;
	case ExperimentPacketPhase::CallbacksComplete: result["phase"].String() = "callbacks_complete"; break;
	}
	return result;
}

class HexWriter final : public IBinaryWriter
{
public:
	std::string hex;
	int write(const std::byte * data, unsigned size) override
	{
		constexpr char digits[] = "0123456789abcdef";
		for(unsigned index = 0; index < size; ++index)
		{
			const auto value = static_cast<unsigned char>(data[index]);
			hex += digits[value >> 4];
			hex += digits[value & 15];
		}
		return size;
	}
};

JsonNode requestRecord(const CPackForServer & pack, PlayerColor player)
{
	HexWriter writer;
	BinarySerializer serializer(&writer);
	serializer & static_cast<const CPack *>(&pack);
	JsonNode result;
	result["kind"].String() = "request";
	result["player"].Integer() = player.getNum();
	result["request_id"].Integer() = pack.requestID;
	result["pack_type"].Integer() = CTypeList::getInstance().getTypeID(&pack);
	result["pack"].String() = writer.hex;
	const int learner = experimentBuiltinLearner();
	if(learner >= 0 && player == PlayerColor(1 - learner))
	{
		if(transactionActor != "client")
			throw std::runtime_error("Native opponent request has an adapter transaction");
		result["source"].String() = "native_opponent";
	}
	else
		result["source"].String() = transactionActor.starts_with("player") ? "adapter" : "automatic";
	return result;
}
}

ContextualEvidence * ContextualEvidence::current()
{
	return dynamic_cast<ContextualEvidence *>(experimentEvidenceHooks().get());
}

bool ContextualEvidence::replaying()
{
	const auto * value = current();
	return value && value->replay;
}

bool ContextualEvidence::cancelled()
{
	const auto * value = current();
	if(!value)
		return false;
	std::lock_guard lock(value->mutex);
	return value->replay ? value->stopping : value->sealed;
}

ContextualEvidence & ContextualEvidence::live(const std::string & episode, bool town, bool armies, bool heroes, bool management, bool dialogs, bool expanded, bool ownArmies)
{
	if(ownArmies && !expanded)
		throw std::runtime_error("Passive army information requires the v8 interface");
	if(auto * existing = current())
	{
		const auto & existingMetadata = existing->metadata;
		if(existing->metadata["episode"].String() != episode
			|| existingMetadata["controllers"] != experimentBuiltinControllers()
			|| existingMetadata["army_information"] != (ownArmies ? JsonNode(OWN_ARMY_INFORMATION) : JsonNode())
			|| existing->metadata["interface"].String() != (expanded ? "adventure-capabilities-v1" : dialogs ? "adventure-dialogs-v1" : management ? "adventure-army-management-v1" : heroes ? "adventure-heroes-v1" : armies ? "adventure-armies-v1" : town ? "adventure-town-v1" : "adventure-windows-v1"))
			throw std::runtime_error("Contextual replay identity changed during initialization");
		return *existing;
	}
	recorder.reset(new ContextualEvidence(episode, town, armies, heroes, management, dialogs, expanded, ownArmies));
	setExperimentEvidenceHooks(recorder);
	return *recorder;
}

ContextualEvidence::ContextualEvidence(const std::string & episode, bool town, bool armies, bool heroes, bool management, bool dialogs, bool expanded, bool ownArmies)
{
	const char * directory = std::getenv("VCMI_EXPERIMENT_DIR");
	if(!directory || !*directory)
		throw std::runtime_error("Contextual evidence requires VCMI_EXPERIMENT_DIR");
	output.open(std::string(directory) + "/contextual-replay.jsonl", std::ios::out | std::ios::trunc);
	if(!output)
		throw std::runtime_error("Cannot create contextual replay ledger");
	metadata["kind"].String() = "init";
	builtin = experimentBuiltinLearner() >= 0;
	metadata["schema"].String() = builtin ? BUILTIN_SCHEMA : SCHEMA;
	if(builtin)
		metadata["controllers"] = experimentBuiltinControllers();
	metadata["episode"].String() = episode;
	metadata["interface"].String() = expanded ? "adventure-capabilities-v1" : dialogs ? "adventure-dialogs-v1" : management ? "adventure-army-management-v1" : heroes ? "adventure-heroes-v1" : armies ? "adventure-armies-v1" : town ? "adventure-town-v1" : "adventure-windows-v1";
	metadata["protocol_version"].Integer() = expanded ? 10 : dialogs ? 9 : management ? 8 : heroes ? 7 : armies ? 6 : town ? 5 : 4;
	metadata["history_schema"].String() = expanded ? "heroes-player-history-v8" : dialogs ? "heroes-player-history-v7" : management ? "heroes-player-history-v6" : heroes ? "heroes-player-history-v5" : armies ? "heroes-player-history-v4" : town ? "heroes-player-history-v3" : "heroes-player-history-v2";
	metadata["action_schema"].String() = expanded ? "heroes-context-actions-v8" : dialogs ? "heroes-context-actions-v7" : management ? "heroes-context-actions-v6" : heroes ? "heroes-context-actions-v5" : armies ? "heroes-context-actions-v4" : town ? "heroes-context-actions-v3" : "heroes-context-actions-v2";
	if(ownArmies)
		metadata["army_information"].String() = OWN_ARMY_INFORMATION;
	output << line(metadata) << '\n';
	output.flush();
	if(!output)
		throw std::runtime_error("Cannot write contextual replay header");
}

void ContextualEvidence::count(JsonNode & counts, const JsonNode & record) const
{
	const auto & kind = record["kind"].String();
	if(kind == "enter")
	{
		++counts["transactions"].Integer();
		if(record["actor"].String() == "client" && record["operation"].String() == "packet")
			++counts["client_packets"].Integer();
	}
	else if(kind == "message") ++counts["messages"].Integer();
	else if(kind == "response") ++counts["responses"].Integer();
	else if(kind == "request")
	{
		++counts["requests"].Integer();
		if(builtin && record["source"].String() == "native_opponent")
			++counts["native_opponent_requests"].Integer();
	}
}

ContextualEvidence::ContextualEvidence(const std::vector<JsonNode> & tape, bool prefix)
	: prefix(prefix), replay(true)
{
	if(tape.empty() || tape.front()["kind"].String() != "init"
		|| (tape.front()["schema"].String() != SCHEMA && tape.front()["schema"].String() != BUILTIN_SCHEMA))
		throw std::runtime_error("Missing current contextual replay header");
	metadata = tape.front();
	builtin = metadata["schema"].String() == BUILTIN_SCHEMA;
	if(builtin != (experimentBuiltinLearner() >= 0)
		|| tape.front()["controllers"] != experimentBuiltinControllers())
		throw std::runtime_error("Contextual replay controller mapping differs from explicit mode");
	const bool expanded = metadata["interface"].String() == "adventure-capabilities-v1";
	const bool dialogs = metadata["interface"].String() == "adventure-dialogs-v1";
	const bool management = metadata["interface"].String() == "adventure-army-management-v1";
	const bool heroes = metadata["interface"].String() == "adventure-heroes-v1";
	const bool armies = metadata["interface"].String() == "adventure-armies-v1";
	const bool town = metadata["interface"].String() == "adventure-town-v1";
	if(metadata.Struct().count("army_information")
		&& (!expanded || !metadata["army_information"].isString() || metadata["army_information"].String() != OWN_ARMY_INFORMATION))
		throw std::runtime_error("Unsupported recorded passive army information contract");
	if((!expanded && !dialogs && !management && !heroes && !armies && !town && metadata["interface"].String() != "adventure-windows-v1")
		|| metadata["protocol_version"].Integer() != (expanded ? 10 : dialogs ? 9 : management ? 8 : heroes ? 7 : armies ? 6 : town ? 5 : 4)
		|| metadata["history_schema"].String() != (expanded ? "heroes-player-history-v8" : dialogs ? "heroes-player-history-v7" : management ? "heroes-player-history-v6" : heroes ? "heroes-player-history-v5" : armies ? "heroes-player-history-v4" : town ? "heroes-player-history-v3" : "heroes-player-history-v2")
		|| metadata["action_schema"].String() != (expanded ? "heroes-context-actions-v8" : dialogs ? "heroes-context-actions-v7" : management ? "heroes-context-actions-v6" : heroes ? "heroes-context-actions-v5" : armies ? "heroes-context-actions-v4" : town ? "heroes-context-actions-v3" : "heroes-context-actions-v2"))
		throw std::runtime_error("Inconsistent contextual replay interface versions");
	for(const auto * key : {"transactions", "messages", "responses", "requests", "client_packets"})
		loaded[key].Integer() = checked[key].Integer() = 0;
	if(builtin)
		loaded["native_opponent_requests"].Integer() = checked["native_opponent_requests"].Integer() = 0;
	std::optional<std::pair<std::string, std::string>> transaction;
	bool closed = false;
	for(size_t index = 1; index < tape.size(); ++index)
	{
		const auto & item = tape[index];
		if(item["seq"].Integer() != static_cast<int64_t>(index - 1) || closed)
			throw std::runtime_error("Contextual replay sequence gap or records after close");
		const auto & kind = item["kind"].String();
		if(kind == "enter")
		{
			if(transaction) throw std::runtime_error("Nested contextual transaction");
			transaction = std::pair{item["actor"].String(), item["operation"].String()};
		}
		else if(kind == "leave")
		{
			if(!transaction || *transaction != std::pair{item["actor"].String(), item["operation"].String()})
				throw std::runtime_error("Unmatched contextual transaction end");
			transaction.reset();
		}
		else if(kind == "close")
		{
			if(transaction || item["barrier_id"].Integer() != 1)
				throw std::runtime_error("Contextual close is not after the terminal transaction");
			closed = true;
		}
		else if(kind != "message" && kind != "response" && kind != "request" && kind != "packet_complete")
			throw std::runtime_error("Unknown contextual replay record kind");
		else if(!transaction)
			throw std::runtime_error("Contextual payload outside transaction");
		count(loaded, item);
		records.push_back(item);
	}
	openFinalTransaction = transaction.has_value();
	if((!prefix && (!closed || transaction)) || (prefix && closed))
		throw std::runtime_error("Contextual replay boundary does not match natural/cutoff mode");
}

// Registry ownership is released explicitly after replay workers have joined,
// or by process teardown. Looking up that owner here would resurrect the
// shared_ptr whose last release is already destroying this object.
ContextualEvidence::~ContextualEvidence() = default;

void ContextualEvidence::finishNativeSession()
{
	std::lock_guard lock(mutex);
	if(replay || recorder.get() != this)
		throw std::logic_error("Native session can close only its detached live recorder");
	output.flush();
	if(!output) throw std::runtime_error("Cannot flush final native session evidence");
	output.close();
	// The caller retains a shared owner throughout this call. Release the
	// module's sticky live owner so no prior episode ledger/state is retained.
	recorder.reset();
}

void ContextualEvidence::append(JsonNode record)
{
	if(sealed) throw std::runtime_error("Contextual evidence is sealed at its terminal barrier");
	record["seq"].Integer() = nextSequence++;
	output << line(record) << '\n';
	output.flush();
	if(!output)
		throw std::runtime_error("Cannot flush contextual replay ledger");
}

void ContextualEvidence::advance()
{
	count(checked, records.at(cursor));
	++cursor;
	if(cursor == records.size()) stopping = true;
	changed.notify_all();
	if(stopping && prefix) throw ContextualPrefixFinished();
}

void ContextualEvidence::fail(const std::string & message)
{
	if(error.empty()) error = message;
	stopping = true;
	changed.notify_all();
}

void ContextualEvidence::drainAutomatic()
{
	while(cursor < records.size() && records[cursor]["kind"].String() == "request"
		&& (records[cursor]["source"].String() == "automatic"
			|| (builtin && records[cursor]["source"].String() == "native_opponent")))
	{
		if(!automaticRequest) throw std::runtime_error("Automatic replay request handler missing");
		automaticRequest(records[cursor]);
		advance();
	}
}

void ContextualEvidence::expect(JsonNode record)
{
	drainAutomatic();
	if(cursor == records.size())
	{
		if(prefix) { stopping = true; return; }
		throw std::runtime_error("Regenerated contextual record beyond recorded stream");
	}
	record["seq"].Integer() = cursor;
	if(record != records[cursor])
		throw std::runtime_error("Contextual replay differs at record " + std::to_string(cursor)
			+ ": expected " + line(records[cursor]) + "; actual " + line(record));
	advance();
}

void ContextualEvidence::rejectSealedEvent(const std::string & actor, const std::string & operation) const
{
	if(!replay && sealed)
	{
		// A late Nullkiller callback may arrive after the final packet released
		// the event mutex. Cancel before entry, allocation of a request ID or
		// sending anything. Never suppress an error inside an entered event.
		if(builtin && actor == "client" && operation == "request")
			throw ExperimentTerminalFinished();
		throw std::runtime_error("Contextual evidence is sealed at its terminal barrier");
	}
}

void ContextualEvidence::beforeEvent(const std::string & actor, const std::string & operation)
{
	std::unique_lock lock(mutex);
	if(!replay)
	{
		rejectSealedEvent(actor, operation);
		return;
	}
	const bool ready = changed.wait_for(lock, std::chrono::seconds(30), [&]
	{
		return stopping || (cursor < records.size() && records[cursor]["kind"].String() == "enter"
			&& records[cursor]["actor"].String() == actor && records[cursor]["operation"].String() == operation);
	});
	if(!ready) fail("Contextual transaction scheduler timed out at record " + std::to_string(cursor));
	if(stopping) throw std::runtime_error(error.empty() ? "Contextual replay prefix finished" : error);
}

void ContextualEvidence::eventEntered(const std::string & actor, const std::string & operation)
{
	std::lock_guard lock(mutex);
	rejectSealedEvent(actor, operation);
	transactionActor = actor;
	JsonNode record;
	record["kind"].String() = "enter";
	record["actor"].String() = actor;
	record["operation"].String() = operation;
	record["packet"] = position();
	if(replay) expect(record); else append(record);
}

void ContextualEvidence::eventLeaving(const std::string & actor, const std::string & operation)
{
	std::lock_guard lock(mutex);
	try
	{
		JsonNode record;
		record["kind"].String() = "leave";
		record["actor"].String() = actor;
		record["operation"].String() = operation;
		if(replay) expect(record); else append(record);
		if(pendingClose)
		{
			if(replay) expect(*pendingClose); else append(*pendingClose);
			pendingClose.reset();
			sealed = true;
		}
	}
	catch(const ContextualPrefixFinished &) {}
	catch(const std::exception & caught)
	{
		fail(caught.what());
		if(!replay) reportExperimentFailure(caught.what());
	}
	transactionActor.clear();
}

void ContextualEvidence::nativeRequest(const CPackForServer & pack, PlayerColor player)
{
	std::lock_guard lock(mutex);
	auto record = requestRecord(pack, player);
	if(replay) expect(record); else append(record);
}

void ContextualEvidence::callbacksComplete(const CPackForClient &, ExperimentReceiptPosition value)
{
	std::lock_guard lock(mutex);
	JsonNode record;
	record["kind"].String() = "packet_complete";
	record["index"].Integer() = value.index;
	record["type"].Integer() = value.type;
	record["completed"].Integer() = value.completed;
	if(replay) expect(record); else append(record);
}

void ContextualEvidence::terminalBarrierComplete(const ExperimentTerminalBarrier & value)
{
	std::lock_guard lock(mutex);
	if(!actualTerminalBarrier.isNull()) throw std::runtime_error("Repeated contextual terminal barrier");
	actualTerminalBarrier["schema"].String() = "heroes-contextual-terminal-barrier-v1";
	actualTerminalBarrier["kind"].String() = "client";
	actualTerminalBarrier["episode"].String() = value.episode;
	actualTerminalBarrier["interface"].String() = value.interface;
	actualTerminalBarrier["barrier_id"].Integer() = value.barrierId;
	actualTerminalBarrier["origin"].String() = "request";
	actualTerminalBarrier["request"]["player"].Integer() = value.requestPlayer.getNum();
	actualTerminalBarrier["request"]["id"].Integer() = value.requestID;
	actualTerminalBarrier["request"]["type"].Integer() = value.requestType;
	actualTerminalBarrier["client_packets"].Integer() = value.clientPackets;
	actualTerminalBarrier["outcomes"].Vector() = {JsonNode(value.redWon ? "win" : "loss"), JsonNode(value.redWon ? "loss" : "win")};
	actualTerminalBarrier["callbacks_complete"].Bool() = true;
	actualTerminalBarrier["observed_count"].Integer() = experimentReceiptPosition().completed;
	if(builtin)
	{
		actualTerminalBarrier["controllers"] = metadata["controllers"];
		actualTerminalBarrier["native_terminal_callbacks"].Vector() = {JsonNode(0), JsonNode(1)};
		actualTerminalBarrier["explorer_terminal_callbacks"] = metadata["controllers"]["explorer_players"];
	}
	JsonNode record;
	record["kind"].String() = "close";
	record["barrier_id"].Integer() = value.barrierId;
	record["client_packets"].Integer() = value.clientPackets;
	pendingClose = record;
}

void ContextualEvidence::message(int player, const JsonNode & value)
{
	std::lock_guard lock(mutex);
	JsonNode record;
	record["kind"].String() = "message";
	record["player"].Integer() = player;
	record["message"] = value;
	if(replay) expect(record); else append(record);
}

void ContextualEvidence::response(int player, const JsonNode & value)
{
	std::lock_guard lock(mutex);
	JsonNode record;
	record["kind"].String() = "response";
	record["player"].Integer() = player;
	record["response"] = value;
	if(replay) expect(record); else append(record);
}

JsonNode ContextualEvidence::replayResponse(int player)
{
	const std::string actor = "player" + std::to_string(player);
	ExperimentEventGuard guard(actor.c_str(), "receive");
	std::lock_guard lock(mutex);
	if(cursor >= records.size() || records[cursor]["kind"].String() != "response"
		|| records[cursor]["player"].Integer() != player)
		throw std::runtime_error("Expected one policy response inside receive transaction");
	auto result = records[cursor]["response"];
	advance();
	return result;
}

void ContextualEvidence::setAutomaticRequest(std::function<void(const JsonNode &)> callback)
{
	automaticRequest = std::move(callback);
}

std::optional<std::pair<std::string, std::string>> ContextualEvidence::nextTransaction()
{
	std::lock_guard lock(mutex);
	if(stopping) return std::nullopt;
	if(cursor < records.size() && records[cursor]["kind"].String() == "enter")
		return std::pair{records[cursor]["actor"].String(), records[cursor]["operation"].String()};
	return std::nullopt;
}

void ContextualEvidence::automaticTransaction()
{
	ExperimentEventGuard guard("client", "request");
	std::lock_guard lock(mutex);
	drainAutomatic();
}

void ContextualEvidence::waitForProgress()
{
	std::unique_lock lock(mutex);
	const auto previous = cursor;
	if(!changed.wait_for(lock, std::chrono::seconds(30), [&]
	{
		return cursor != previous || stopping || (cursor < records.size()
			&& records[cursor]["kind"].String() == "enter" && records[cursor]["actor"].String() == "client");
	}))
		fail("Contextual replay made no progress at record " + std::to_string(cursor));
}

void ContextualEvidence::stop()
{
	std::lock_guard lock(mutex);
	stopping = true;
	changed.notify_all();
}

void ContextualEvidence::check() const
{
	std::lock_guard lock(mutex);
	if(!error.empty()) throw std::runtime_error(error);
}

JsonNode ContextualEvidence::summary(uint64_t regeneratedPackets) const
{
	std::lock_guard lock(mutex);
	if(!error.empty()) throw std::runtime_error(error);
	if(cursor != records.size()) throw std::runtime_error("Unconsumed contextual replay tail");
	JsonNode result;
	if(builtin)
		result["controllers"] = metadata["controllers"];
	result["loaded"] = loaded;
	result["checked"] = checked;
	for(const auto & [key, value] : loaded.Struct())
	{
		const auto remaining = value.Integer() - checked[key].Integer();
		result["unchecked"][key].Integer() = remaining;
		if(remaining != 0) throw std::runtime_error("Unchecked contextual replay records");
	}
	const auto packets = checked["client_packets"].Integer();
	if(regeneratedPackets < static_cast<uint64_t>(packets) || (!prefix && regeneratedPackets != static_cast<uint64_t>(packets)))
		throw std::runtime_error("Regenerated packet range does not match contextual boundary");
	result["regenerated_server_packets_after_prefix"].Integer() = regeneratedPackets - packets;
	result["open_final_transaction"].Bool() = openFinalTransaction;
	result["captured_prefix_only"].Bool() = prefix;
	result["server_verified"].Bool() = true;
	result["boundary"]["kind"].String() = prefix ? "cutoff" : "natural";
	result["boundary"]["barrier_id"] = prefix ? JsonNode() : JsonNode(1);
	result["boundary"]["client_packets"].Integer() = packets;
	if(prefix != actualTerminalBarrier.isNull())
		throw std::runtime_error("Regenerated terminal barrier does not match replay boundary");
	result["terminal_barrier"] = actualTerminalBarrier;
	result["final_client_position"] = position();
	return result;
}
}
