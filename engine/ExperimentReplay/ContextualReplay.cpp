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

/// Recorded opponent requests are external inputs. Never rerun Nullkiller or
/// invent its decisions, observations or memories during exact verification.
class PassiveOpponent final : public CGameInterface
{
public:
	PassiveOpponent() { human = true; }
	void activeStack(const BattleID &, const CStack *) override {}
	void yourTacticPhase(const BattleID &, int) override {}
	void heroGotLevel(const CGHeroInstance *, PrimarySkill, std::vector<SecondarySkill> &, QueryID) override {}
	void commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) override {}
	void showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override {}
	void showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) override {}
	void showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) override {}
	void showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) override {}
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &) override { return std::nullopt; }
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

void ContextualReplay::configureControllers(const JsonNode & header)
{
	unsetenv("EXPLORER_RL_PLAYER");
	if(header["schema"].String() == explorer_rl::ContextualEvidence::BUILTIN_SCHEMA)
	{
		const auto & players = header["controllers"]["explorer_players"];
		if(header["interface"].String() != "adventure-dialogs-v1" || !players.isVector()
			|| players.Vector().size() != 1 || players[0].getType() != JsonNode::JsonType::DATA_INTEGER
			|| (players[0].Integer() != 0 && players[0].Integer() != 1))
			throw std::runtime_error("Invalid recorded built-in controller configuration");
		setenv("EXPLORER_RL_BUILTIN_OPPONENT", "Nullkiller", 1);
		setenv("EXPLORER_RL_PLAYERS", players[0].Integer() == 0 ? "0" : "1", 1);
		setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
		setenv("EXPLORER_RL_INTERFACE", "adventure-dialogs-v1", 1);
		if(header["controllers"] != experimentBuiltinControllers())
			throw std::runtime_error("Recorded built-in controller fields differ from the supported mapping");
	}
	else
	{
		if(header["schema"].String() != explorer_rl::ContextualEvidence::SCHEMA || !header["controllers"].isNull())
			throw std::runtime_error("Unknown contextual controller mode");
		unsetenv("EXPLORER_RL_BUILTIN_OPPONENT");
		setenv("EXPLORER_RL_PLAYERS", "0,1", 1);
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
	if(tape.empty()) throw std::runtime_error("Missing contextual replay header");
	configureControllers(tape.front());
	auto evidence = std::make_shared<explorer_rl::ContextualEvidence>(tape, prefix);
	const auto & header = evidence->header();
	const int learner = experimentBuiltinLearner();
	setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
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
		if(learner >= 0 && (expected["player"].Integer() != request->player.getNum()
			|| expected["request"].Integer() != request->requestID))
			throw std::runtime_error("Authoritative mixed request metadata differs from its native packet");
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
		client.initExperimentReplay(state, [&](PlayerColor color) -> std::shared_ptr<CGameInterface>
		{
			if(learner >= 0 && color != PlayerColor(learner))
				return std::make_shared<PassiveOpponent>();
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
