#pragma once

#include "../lib/json/JsonNode.h"
#include "../lib/networkPacks/PacksForServer.h"
#include "../lib/serializer/CTypeList.h"

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

/// Check the captured send stream against the independently applied receive
/// stream. A terminal range is supplied only by the verified server replay,
/// never read from the client tape. The exceptional attempt is consumed by the
/// client replay but is never passed to CGameHandler.
inline JsonNode contextualRequestAccounting(const std::vector<JsonNode> & tape,
	const std::vector<JsonNode> & inputs, const JsonNode & terminalInput,
	bool prefix, int learner,
	const std::function<std::unique_ptr<CPack>(const std::string &)> & decode,
	const std::function<bool(const CPack &, const std::string &)> & automaticAllowed)
{
	JsonNode result;
	result["authoritative_inputs"].Integer() = inputs.size();
	result["final_terminal_input"] = terminalInput;
	auto & unapplied = result["terminal_unapplied"].Vector();
	size_t requestIndex = 0;
	std::optional<std::pair<std::string, std::string>> transaction;
	JsonNode receipt;
	bool closed = false;
	for(const auto & item : tape)
	{
		const auto & kind = item["kind"].String();
		if(closed) throw std::runtime_error("Contextual record follows terminal close");
		if(kind == "response" && !unapplied.empty())
			throw std::runtime_error("Learner response follows a terminal-unapplied native attempt");
		if(kind == "enter")
		{
			if(transaction) throw std::runtime_error("Nested contextual transaction");
			transaction = std::pair{item["actor"].String(), item["operation"].String()};
			receipt = item["packet"];
		}
		if(kind == "leave")
		{
			if(!transaction || *transaction != std::pair{item["actor"].String(), item["operation"].String()})
				throw std::runtime_error("Unmatched contextual transaction end");
			transaction.reset();
		}
		if(kind == "close")
		{
			if(transaction || item["barrier_id"].Integer() != 1)
				throw std::runtime_error("Invalid contextual terminal close");
			if(!unapplied.empty() && item["client_packets"] != terminalInput["last_client_packet"])
				throw std::runtime_error("Terminal attempt close differs from verified server boundary");
			closed = true;
		}
		if(kind != "request") continue;
		if(!transaction) throw std::runtime_error("Native request outside a transaction");
		auto decoded = decode(item["pack"].String());
		const auto * request = dynamic_cast<const CPackForServer *>(decoded.get());
		if(!request || item["seq"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| item["player"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| item["request_id"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| item["pack_type"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| item["player"].Integer() != request->player.getNum()
			|| item["request_id"].Integer() != request->requestID
			|| item["pack_type"].Integer() != CTypeList::getInstance().getTypeID(request))
			throw std::runtime_error("Native request metadata does not bind its serialized packet");
		const auto & source = item["source"].String();
		if(source == "adapter")
		{
			if(transaction->first != "player" + std::to_string(request->player.getNum())
				|| (learner >= 0 && request->player != PlayerColor(learner)))
				throw std::runtime_error("Adapter request is not inside its player's transaction");
		}
		else if(source == "native_opponent")
		{
			if(learner < 0 || request->player != PlayerColor(1 - learner) || transaction->first != "client"
				|| (transaction->second != "request" && transaction->second != "packet"))
				throw std::runtime_error("Recorded opponent input names the wrong controller or transaction");
		}
		else if(source != "automatic" || transaction->first != "client"
			|| (learner >= 0 && request->player == PlayerColor(1 - learner))
			|| !automaticAllowed(*decoded, transaction->second))
			throw std::runtime_error("Unsupported automatic request source/type");

		if(requestIndex < inputs.size())
		{
			const auto & expected = inputs[requestIndex++];
			if(item["pack"].String() != expected["pack"].String()
				|| item["pack_type"].Integer() != expected["type"].Integer())
				throw std::runtime_error("Client request bytes/order differ from authoritative server input");
			if(learner >= 0 && (expected["player"].Integer() != request->player.getNum()
				|| expected["request"].Integer() != request->requestID))
				throw std::runtime_error("Authoritative mixed request metadata differs from its native packet");
			continue;
		}

		// Only a single opposite-seat MoveHero, BuildStructure, EndTurn or
		// TradeOnMarketplace sent while the
		// final native outcome packets were consumed may have no server application.
		// The sealed server cannot accept it after the terminal input finishes.
		if(prefix || learner < 0 || inputs.empty() || !unapplied.empty() || terminalInput.isNull()
			|| terminalInput["input_seq"].Integer() != static_cast<int64_t>(inputs.size() - 1)
			|| terminalInput["request_id"] != inputs.back()["request"]
			|| terminalInput["player"] != inputs.back()["player"]
			|| terminalInput["pack_type"] != inputs.back()["type"]
			|| source != "native_opponent" || transaction->first != "client" || transaction->second != "request"
			|| request->player != PlayerColor(1 - learner)
			|| (!dynamic_cast<const MoveHero *>(request) && !dynamic_cast<const BuildStructure *>(request) && !dynamic_cast<const EndTurn *>(request) && !dynamic_cast<const TradeOnMarketplace *>(request))
			|| terminalInput["request_id"].Integer() == std::numeric_limits<uint32_t>::max()
			|| item["request_id"].Integer() != terminalInput["request_id"].Integer() + 1
			|| receipt["index"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| receipt["completed"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| receipt["type"].getType() != JsonNode::JsonType::DATA_INTEGER
			|| receipt["index"] != receipt["completed"] || receipt["phase"].String() != "callbacks_complete"
			|| receipt["completed"].Integer() < terminalInput["first_client_packet"].Integer()
			|| receipt["completed"].Integer() >= terminalInput["last_client_packet"].Integer())
			throw std::runtime_error("Client native request has no verified authoritative application or terminal boundary");
		JsonNode attempt;
		for(const auto * key : {"seq", "player", "request_id", "pack_type", "pack", "source"})
			attempt[key] = item[key];
		attempt["receipt"] = receipt;
		unapplied.push_back(std::move(attempt));
	}
	if(requestIndex != inputs.size())
		throw std::runtime_error("Authoritative server input omitted from contextual request ledger");
	if(!unapplied.empty() && !closed)
		throw std::runtime_error("Terminal-unapplied attempt requires the actual natural close");
	return result;
}
