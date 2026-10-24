#include "StdInc.h"
#include "ContextualReplay.h"

#include "../AI/ExplorerRL/ExplorerRL.h"
#include "../AI/ExplorerRL/ContextualEvidence.h"
#include "../client/Client.h"
#include "../client/CServerHandler.h"
#include "../client/GameInstance.h"
#include "../lib/CConfigHandler.h"
#include "../lib/GameLibrary.h"
#include "../lib/gameState/CGameState.h"
#include "../lib/networkPacks/PacksForServer.h"
#include "../lib/serializer/BinarySerializer.h"
#include "../lib/serializer/BinaryDeserializer.h"
#include "../lib/serializer/RegisterTypes.h"
#include "../lib/GameSettings.h"
#include "../lib/RiverHandler.h"
#include "../lib/RoadHandler.h"
#include "../lib/TerrainHandler.h"
#include "../lib/entities/hero/CHero.h"
#include "../lib/mapObjects/ObjectTemplate.h"
#include "../lib/mapping/CMapInfo.h"
#include "../lib/mapping/CCastleEvent.h"
#include "../lib/rmg/CMapGenOptions.h"
#include "../server/ExperimentReplay.h"

namespace
{
class Writer final : public IBinaryWriter
{
public:
	std::vector<std::byte> bytes;
	int write(const std::byte * data, unsigned size) override
	{
		bytes.insert(bytes.end(), data, data + size);
		return size;
	}
};

class Reader final : public IBinaryReader
{
	const std::vector<std::byte> & bytes;
	size_t cursor = 0;
public:
	explicit Reader(const std::vector<std::byte> & value) : bytes(value) {}
	int read(std::byte * data, unsigned size) override
	{
		if(size > bytes.size() - cursor) throw std::runtime_error("Replay initial state ended early");
		std::copy_n(bytes.begin() + cursor, size, data);
		cursor += size;
		return size;
	}
	bool complete() const { return cursor == bytes.size(); }
};

/// Battle choices are external, server-validated inputs just as in legacy
/// replay. Only adventure callbacks are replayed here. This avoids rerunning
/// stochastic BattleAI or manufacturing replacement battle choices.
class PassiveBattle final : public CBattleGameInterface
{
	std::shared_ptr<ExplorerRL> adventure;
public:
	explicit PassiveBattle(std::shared_ptr<ExplorerRL> value) : adventure(std::move(value))
	{
		human = true; // activeStack is synchronous and never unlocks an absent GUI lock.
	}
	void activeStack(const BattleID &, const CStack *) override {}
	void yourTacticPhase(const BattleID &, int) override {}
	void battleStart(const BattleID & id, const CCreatureSet * left, const CCreatureSet * right, int3 tile,
		const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override
	{
		adventure->battleStart(id, left, right, tile, hero1, hero2, side, replayAllowed);
	}
	void battleEnd(const BattleID & id, const BattleResult * result, QueryID query) override
	{
		adventure->battleEnd(id, result, query);
	}
};

/// Automatic requests may initialize the interface or choose a battle action.
/// Movement, turn ending, queries, building and recruitment must be regenerated
/// by the actual adapter and can never be relabelled automatic.
bool automaticAllowed(const CPack & pack, const std::string & operation)
{
	if(operation == "request") return dynamic_cast<const AdvInterfaceReady *>(&pack) != nullptr;
	if(operation == "packet") return dynamic_cast<const MakeAction *>(&pack) != nullptr;
	return false;
}
}

void ContextualReplay::initialize(CGameState & state)
{
	Writer writer;
	BinarySerializer serializer(&writer);
	serializer & state;
	initialState = std::move(writer.bytes);
}

void ContextualReplay::packet(const CPackForClient & pack)
{
	packets.push_back(experiment_replay::encodePack(pack));
}

void ContextualReplay::input(const JsonNode & record)
{
	inputs.push_back(record);
}

JsonNode ContextualReplay::run(const std::vector<JsonNode> & tape, bool prefix)
{
	if(initialState.empty()) throw std::runtime_error("Missing independently regenerated initial client state");
	auto evidence = std::make_shared<explorer_rl::ContextualEvidence>(tape, prefix);
	const auto & header = evidence->header();
	setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
	setenv("EXPLORER_RL_PLAYERS", "0,1", 1);
	setenv("EXPLORER_RL_EPISODE", header["episode"].String().c_str(), 1);
	setenv("EXPLORER_RL_INTERFACE", header["interface"].String().c_str(), 1);
	unsetenv("VCMI_EXPERIMENT_REPLAY");
	setExperimentEvidenceHooks(evidence);
	{
		Settings session = settings.write["session"];
		session["headless"].Bool() = true;
		session["spectate"].Bool() = false;
		session["onlyai"].Bool() = true;
		session["testmap"].String() = "contextual-replay";
	}

	Reader reader(initialState);
	auto state = std::make_shared<CGameState>();
	state->preInit(LIBRARY);
	BinaryDeserializer deserializer(&reader);
	deserializer.version = ESerializationVersion::CURRENT;
	deserializer.loadingGamestate = true;
	deserializer.cb = state.get();
	deserializer & *state;
	if(!reader.complete()) throw std::runtime_error("Trailing replay initial-state bytes");

	// Check every recorded native request against the authoritative receive
	// stream before allowing any request bytes to act as external battle input.
	size_t requestIndex = 0;
	std::optional<std::pair<std::string, std::string>> transaction;
	for(const auto & item : tape)
	{
		const auto & kind = item["kind"].String();
		if(kind == "enter") transaction = std::pair{item["actor"].String(), item["operation"].String()};
		if(kind == "leave") transaction.reset();
		if(kind != "request") continue;
		if(requestIndex >= inputs.size() || !transaction)
			throw std::runtime_error("Client native request has no authoritative input");
		const auto & expected = inputs[requestIndex++];
		if(item["pack"].String() != expected["pack"].String()
			|| item["pack_type"].Integer() != expected["type"].Integer())
			throw std::runtime_error("Client request bytes/order differ from authoritative server input");
		auto decoded = experiment_replay::decodePack(experiment_replay::fromHex(item["pack"].String()), state.get());
		const auto * request = dynamic_cast<const CPackForServer *>(decoded.get());
		if(!request || item["player"].Integer() != request->player.getNum()
			|| item["request_id"].Integer() != request->requestID)
			throw std::runtime_error("Native request metadata does not bind its serialized packet");
		const auto & source = item["source"].String();
		if(source == "adapter")
		{
			if(transaction->first != "player" + std::to_string(request->player.getNum()))
				throw std::runtime_error("Adapter request is not inside its player's transaction");
		}
		else if(source != "automatic" || transaction->first != "client" || !automaticAllowed(*decoded, transaction->second))
			throw std::runtime_error("Unsupported automatic request source/type");
	}
	if(requestIndex != inputs.size())
		throw std::runtime_error("Authoritative server input omitted from contextual request ledger");

	GAME = std::make_unique<GameInstance>();
	GAME->server().client = std::make_unique<CClient>();
	auto & client = *GAME->server().client;
	evidence->setAutomaticRequest([&client](const JsonNode & item)
	{
		auto decoded = experiment_replay::decodePack(experiment_replay::fromHex(item["pack"].String()), &client.gameInfo());
		const auto * request = dynamic_cast<const CPackForServer *>(decoded.get());
		if(!request) throw std::runtime_error("Automatic request is not a native server packet");
		client.acceptExperimentReplayAutomaticRequest(*request);
	});
	std::map<PlayerColor, std::shared_ptr<ExplorerRL>> interfaces;
	auto finish = [&]
	{
		evidence->check();
		for(const auto & [color, interface] : interfaces) interface->finish();
		const auto result = evidence->summary(packets.size());
		GAME.reset();
		setExperimentEvidenceHooks(nullptr);
		return result;
	};
	try
	{
		client.initExperimentReplay(state, [&](PlayerColor color)
		{
			auto interface = std::make_shared<ExplorerRL>();
			interfaces[color] = interface;
			return interface;
		});
		for(const auto & [color, interface] : interfaces)
			client.installNewBattleInterface(std::make_shared<PassiveBattle>(interface), color, false);
		size_t packetIndex = 0;
		while(!explorer_rl::ContextualEvidence::cancelled())
		{
			evidence->check();
			const auto next = evidence->nextTransaction();
			if(next && next->first == "client" && next->second == "packet")
			{
				if(packetIndex >= packets.size()) throw std::runtime_error("Contextual receipt has no regenerated server packet");
				auto decoded = experiment_replay::decodePack(packets[packetIndex++], &client.gameInfo());
				auto * packet = dynamic_cast<CPackForClient *>(decoded.get());
				if(!packet) throw std::runtime_error("Regenerated client packet has wrong direction");
				client.handlePack(*packet);
			}
			else if(next && next->first == "client" && next->second == "request")
				evidence->automaticTransaction();
			else
				evidence->waitForProgress();
		}
		return finish();
	}
	catch(const explorer_rl::ContextualPrefixFinished &)
	{
		return finish();
	}
	catch(...)
	{
		evidence->stop();
		for(const auto & [color, interface] : interfaces) interface->finish();
		GAME.reset();
		setExperimentEvidenceHooks(nullptr);
		throw;
	}
}
