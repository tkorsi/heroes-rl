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

ContextualEvidence & ContextualEvidence::live(const std::string & episode, bool town, bool armies)
{
	if(auto * existing = current())
	{
		if(existing->metadata["episode"].String() != episode
			|| existing->metadata["interface"].String() != (armies ? "adventure-armies-v1" : town ? "adventure-town-v1" : "adventure-windows-v1"))
			throw std::runtime_error("Contextual replay identity changed during initialization");
		return *existing;
	}
	recorder.reset(new ContextualEvidence(episode, town, armies));
	setExperimentEvidenceHooks(recorder);
	return *recorder;
}

ContextualEvidence::ContextualEvidence(const std::string & episode, bool town, bool armies)
{
	const char * directory = std::getenv("VCMI_EXPERIMENT_DIR");
	if(!directory || !*directory)
		throw std::runtime_error("Contextual evidence requires VCMI_EXPERIMENT_DIR");
	output.open(std::string(directory) + "/contextual-replay.jsonl", std::ios::out | std::ios::trunc);
	if(!output)
		throw std::runtime_error("Cannot create contextual replay ledger");
	metadata["kind"].String() = "init";
	metadata["schema"].String() = SCHEMA;
	metadata["episode"].String() = episode;
	metadata["interface"].String() = armies ? "adventure-armies-v1" : town ? "adventure-town-v1" : "adventure-windows-v1";
	metadata["protocol_version"].Integer() = armies ? 6 : town ? 5 : 4;
	metadata["history_schema"].String() = armies ? "heroes-player-history-v4" : town ? "heroes-player-history-v3" : "heroes-player-history-v2";
	metadata["action_schema"].String() = armies ? "heroes-context-actions-v4" : town ? "heroes-context-actions-v3" : "heroes-context-actions-v2";
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
	else if(kind == "request") ++counts["requests"].Integer();
}

ContextualEvidence::ContextualEvidence(const std::vector<JsonNode> & tape, bool prefix)
	: prefix(prefix), replay(true)
{
	if(tape.empty() || tape.front()["kind"].String() != "init" || tape.front()["schema"].String() != SCHEMA)
		throw std::runtime_error("Missing current contextual replay header");
	metadata = tape.front();
	const bool armies = metadata["interface"].String() == "adventure-armies-v1";
	const bool town = metadata["interface"].String() == "adventure-town-v1";
	if((!armies && !town && metadata["interface"].String() != "adventure-windows-v1")
		|| metadata["protocol_version"].Integer() != (armies ? 6 : town ? 5 : 4)
		|| metadata["history_schema"].String() != (armies ? "heroes-player-history-v4" : town ? "heroes-player-history-v3" : "heroes-player-history-v2")
		|| metadata["action_schema"].String() != (armies ? "heroes-context-actions-v4" : town ? "heroes-context-actions-v3" : "heroes-context-actions-v2"))
		throw std::runtime_error("Inconsistent contextual replay interface versions");
	for(const auto * key : {"transactions", "messages", "responses", "requests", "client_packets"})
		loaded[key].Integer() = checked[key].Integer() = 0;
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
		&& records[cursor]["source"].String() == "automatic")
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

void ContextualEvidence::beforeEvent(const std::string & actor, const std::string & operation)
{
	std::unique_lock lock(mutex);
	if(!replay)
	{
		if(sealed) throw std::runtime_error("Contextual evidence is sealed at its terminal barrier");
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
