/*
 * ReplayMain.cpp, part of the Heroes RL experiment tools for VCMI
 *
 * License: GNU General Public License v2.0 or later
 *
 */
#include "StdInc.h"
#ifdef HEROES_CONTEXTUAL_REPLAY
#include "ContextualReplay.h"
#endif

#include "../AI/ExplorerRL/VisibleWorld.h"
#include "../server/CGameHandler.h"
#include "../server/ExperimentReplay.h"
#include "../server/IGameServer.h"

#include "../lib/CConfigHandler.h"
#include "../lib/CConsoleHandler.h"
#include "../lib/GameConstants.h"
#include "../lib/GameLibrary.h"
#include "../lib/LoadProgress.h"
#include "../lib/StartInfo.h"
#include "../lib/battle/BattleInfo.h"
#include "../lib/VCMIDirs.h"
#include "../lib/callback/CCallback.h"
#include "../lib/gameState/CGameState.h"
#include "../lib/json/JsonNode.h"
#include "../lib/logging/CBasicLogConfigurator.h"
#include "../lib/mapObjects/CGObjectInstance.h"
#include "../lib/mapping/CMap.h"
#include "../lib/networkPacks/NetPacksBase.h"
#include "../lib/serializer/CTypeList.h"
#include "../lib/serializer/ESerializationVersion.h"

#include <fstream>

/// Replays the inputs an experiment's server recorded and stops at the first
/// divergence from the recorded outcomes, sent-pack digests, state digests or
/// the players' recorded observation digests.
///
/// vcmireplay --log <engine>/replay.jsonl --observations <engine> --report <report.json>
///            [--trace <trace.jsonl>] [--prefix] [--allow-platform-mismatch]
///
/// --prefix accepts a log the environment cut short by stopping the engine:
/// its records are replayed in full, and observations the clients recorded
/// after the last replayed pack are counted instead of required. Every loaded
/// observation is either checked or counted: a record must name a client pack
/// position of at least 1, and positions may not decrease within a player's file.
namespace
{
using namespace experiment_replay;

/// Where and why a replay stopped matching its recording.
struct Divergence : std::runtime_error
{
	JsonNode details;

	Divergence(const std::string & stage, const std::string & field, const std::string & expected, const std::string & actual)
		: std::runtime_error(stage + ": " + field + " differs")
	{
		details["stage"].String() = stage;
		details["field"].String() = field;
		details["expected"].String() = expected;
		details["actual"].String() = actual;
	}
};

void expectEqual(const std::string & stage, const std::string & field, const std::string & expected, const std::string & actual)
{
	if(expected != actual)
		throw Divergence(stage, field, expected, actual);
}

/// The sent-pack chain, naming the first pack that differs when it does not match.
void expectChain(const std::string & stage, const JsonNode & record, const Outgoing & outgoing)
{
	if(record["chain"].String() == digestHex(outgoing.chain))
		return;
	const auto & recorded = record["sent"].Vector();
	for(size_t index = 0; index < std::max(recorded.size(), outgoing.sent.size()); ++index)
	{
		const std::string expected = index < recorded.size()
			? std::to_string(recorded[index][0].Integer()) + ":" + recorded[index][1].String() : "none";
		const std::string actual = index < outgoing.sent.size()
			? std::to_string(outgoing.sent[index].first) + ":" + digestHex(outgoing.sent[index].second) : "none";
		expectEqual(stage, "sent pack " + std::to_string(index) + " (type:digest)", expected, actual);
	}
	throw Divergence(stage, "chain", record["chain"].String(), digestHex(outgoing.chain));
}

/// The logical-state digest, naming the first part that differs when it does not match.
void expectState(const std::string & stage, const JsonNode & record, CGameHandler & gameHandler)
{
	const auto actual = digestHex(stateDigest(gameHandler));
	if(record["digest"].String() == actual)
		return;
	for(const auto & [name, digest] : stateDigestParts(gameHandler))
		expectEqual(stage, std::string("digest part ") + name, record["parts"][name].String(), digestHex(digest));
	throw Divergence(stage, "digest", record["digest"].String(), actual);
}

JsonNode parseLine(const std::string & line, const std::string & name)
{
	JsonParsingSettings parsing;
	parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
	parsing.strict = true;
	return JsonNode(reinterpret_cast<const std::byte *>(line.data()), line.size(), parsing, name);
}

std::vector<JsonNode> readLines(const boost::filesystem::path & path)
{
	std::ifstream input(path.string());
	if(!input)
		throw std::runtime_error("Cannot read " + path.string());
	std::vector<JsonNode> result;
	std::string line;
	while(std::getline(input, line))
		if(!line.empty())
			result.push_back(parseLine(line, path.string()));
	return result;
}

/// One JSON line: VCMI's compact writer keeps formatting newlines in structs
/// with more than one key, and JSON strings escape their own line breaks.
std::string jsonLine(const JsonNode & node)
{
	auto line = node.toCompactString();
	line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return c == '\n' || c == '\r'; }), line.end());
	return line;
}

std::string text(const JsonNode & node)
{
	if(node.isNull())
		return "null";
	if(node.getType() == JsonNode::JsonType::DATA_BOOL)
		return node.Bool() ? "true" : "false";
	if(node.getType() == JsonNode::JsonType::DATA_STRING)
		return node.String();
	if(node.isNumber())
		return std::to_string(node.Integer());
	return jsonLine(node);
}

/// One player capture the client recorded at a position in the pack stream.
struct ObservationRecord
{
	int player;
	std::string kind; ///< capture or message
	int64_t number; ///< visible sequence or message index
	uint16_t packType;
	std::string digest;
};

/// The server side of a replay: answers the game handler like the recorded
/// server did and digests what it sends, as the live recorder did.
class ReplayServer final : public IGameServer
{
public:
	CGameHandler * gameHandler = nullptr;
	Outgoing outgoing;
	int clientConnection = -1;
	std::map<int, std::set<int>> connections;
	std::set<int> hosts;
	EServerState state = EServerState::LOBBY;
	std::function<void(uint16_t)> clientPackApplied;
	std::function<void(const CPackForClient &)> clientPackProduced;
	uint64_t experimentClientPacketCount() const override { return outgoing.clientCount; }
	GameConnectionID experimentClientConnection() const override { return GameConnectionID(static_cast<int8_t>(clientConnection)); }

	void setState(EServerState value) override { state = value; }
	EServerState getState() const override { return state; }
	bool isPlayerHost(const PlayerColor & color) const override { return hosts.count(color.getNum()) != 0; }

	bool hasPlayerAt(PlayerColor player, GameConnectionID connectionID) const override
	{
		const auto connection = connections.find(static_cast<int>(connectionID));
		return connection != connections.end() && connection->second.count(player.getNum());
	}

	bool hasBothPlayersAtSameConnection(PlayerColor left, PlayerColor right) const override
	{
		for(const auto & [connection, players] : connections)
			if(players.count(left.getNum()) && players.count(right.getNum()))
				return true;
		return false;
	}

	void applyPack(CPackForClient & pack) override
	{
		// As CVCMIServer::applyPack: noted, sent, then applied to the server's state.
		const auto before = outgoing.clientCount;
		outgoing.note(pack, -1, clientConnection);
		if(outgoing.clientCount != before && clientPackProduced)
			clientPackProduced(pack);
		gameHandler->gs->apply(pack);
		if(outgoing.clientCount != before && clientPackApplied)
			clientPackApplied(CTypeList::getInstance().getTypeID(&pack));
	}

	void sendPack(CPackForClient & pack, GameConnectionID connectionID) override
	{
		const auto before = outgoing.clientCount;
		outgoing.note(pack, static_cast<int>(connectionID), clientConnection);
		if(outgoing.clientCount != before && clientPackProduced)
			clientPackProduced(pack);
		if(outgoing.clientCount != before && clientPackApplied)
			clientPackApplied(CTypeList::getInstance().getTypeID(&pack));
	}
};

class Replay
{
#ifdef HEROES_CONTEXTUAL_REPLAY
	ContextualReplay * contextual = nullptr;
#endif
	ReplayServer server;
	std::unique_ptr<CGameHandler> gameHandler;
	std::map<int, std::shared_ptr<CCallback>> callbacks; ///< per recorded player, over the replayed state
	std::multimap<uint64_t, ObservationRecord> observations; ///< by client pack position
	std::map<int, int64_t> observationsLoaded;
	std::map<int, int64_t> observationsChecked;
	std::ofstream trace;
	int64_t inputs = 0;
	std::map<std::string, int64_t> boundaries;
	JsonNode terminals;

	uint64_t digestCapture(int player)
	{
		const auto & callback = *callbacks.at(player);
		return explorer_rl::snapshotDigest(explorer_rl::captureVisible(
			explorer_rl::CallbackWorld{callback, PlayerColor(player), callback.getMapSize()}));
	}

	void checkObservations(uint16_t packType)
	{
		const auto position = server.outgoing.clientCount;
		const auto [first, last] = observations.equal_range(position);
		for(auto entry = first; entry != last; ++entry)
		{
			const auto & record = entry->second;
			const std::string stage = "observation p" + std::to_string(record.player) + " " + record.kind + " "
				+ std::to_string(record.number) + " at client pack " + std::to_string(position);
			expectEqual(stage, "pack_type", std::to_string(record.packType), std::to_string(packType));
			expectEqual(stage, "digest", record.digest, digestHex(digestCapture(record.player)));
			++observationsChecked[record.player];
		}
	}

	void writeTrace(int64_t sequence, const char * boundary)
	{
		if(!trace.is_open())
			return;
		const auto & state = gameHandler->gameState();
		JsonNode frame;
		frame["kind"].String() = "frame";
		frame["seq"].Integer() = sequence;
		frame["day"].Integer() = state.getDate(Date::DAY);
		frame["boundary"] = boundary ? JsonNode(boundary) : JsonNode();
		auto & battles = frame["battles"].Vector();
		battles.clear();
		for(const auto & battle : state.currentBattles)
		{
			// Where, and which players' clients took part: only they saw the battle.
			JsonNode entry;
			for(const int64_t value : {int64_t{battle->tile.x}, int64_t{battle->tile.y}, int64_t{battle->tile.z},
					int64_t{battle->getSidePlayer(BattleSide::ATTACKER).getNum()},
					int64_t{battle->getSidePlayer(BattleSide::DEFENDER).getNum()}})
				entry.Vector().emplace_back(value);
			battles.push_back(std::move(entry));
		}
		auto & objects = frame["objects"].Vector();
		for(const auto & object : state.getMap().getObjects())
		{
			if(!object)
				continue;
			const auto position = object->visitablePos();
			JsonNode entry;
			for(const int64_t value : {int64_t{object->id.getNum()}, int64_t{position.x}, int64_t{position.y}, int64_t{position.z},
					int64_t{object->ID.getNum()}, int64_t{object->subID.getNum()}, int64_t{object->getOwner().getNum()}})
				entry.Vector().emplace_back(value);
			objects.push_back(std::move(entry));
		}
		for(const auto & [player, callback] : callbacks)
		{
			const auto snapshot = explorer_rl::captureVisible(explorer_rl::CallbackWorld{*callback, PlayerColor(player), callback->getMapSize()});
			JsonNode view;
			std::string visible;
			for(const auto tile : snapshot.tiles)
				visible += tile ? '1' : '0';
			view["visible"].String() = visible;
			for(const auto & [id, object] : snapshot.objects)
				view["objects"].Vector().emplace_back(id);
			frame["players"][std::to_string(player)] = view;
		}
		trace << jsonLine(frame) << '\n';
	}

	void writeTraceMap()
	{
		if(!trace.is_open())
			return;
		const auto & map = gameHandler->gameState().getMap();
		JsonNode header;
		header["kind"].String() = "map";
		header["width"].Integer() = map.width;
		header["height"].Integer() = map.height;
		header["levels"].Integer() = map.levels();
		auto & terrain = header["terrain"].Vector();
		for(int z = 0; z < map.levels(); ++z)
			for(int y = 0; y < map.height; ++y)
				for(int x = 0; x < map.width; ++x)
					terrain.emplace_back(static_cast<int64_t>(map.getTile(int3(x, y, z)).getTerrainID().getNum()));
		trace << jsonLine(header) << '\n';
	}

public:
	void loadObservations(const boost::filesystem::path & directory)
	{
		for(int player = 0; player < PlayerColor::PLAYER_LIMIT_I; ++player)
		{
			const auto path = directory / ("observations-p" + std::to_string(player) + ".jsonl");
			if(!boost::filesystem::exists(path))
				continue;
			int64_t last = 1; // a capture follows at least the client's first applied pack
			int64_t loaded = 0;
			for(const auto & line : readLines(path))
			{
				const auto & kind = line["kind"].String();
				ObservationRecord record{player, kind, line[kind == "capture" ? "sequence" : "index"].Integer(),
					static_cast<uint16_t>(line["pack_type"].Integer()), line["digest"].String()};
				const std::string stage = "observation p" + std::to_string(player) + " " + kind + " " + std::to_string(record.number);
				if(kind != "capture" && kind != "message")
					throw Divergence(stage, "kind", "capture or message", text(line["kind"]));
				const auto & pack = line["pack"];
				if(pack.getType() != JsonNode::JsonType::DATA_INTEGER || pack.Integer() < last)
					throw Divergence(stage, "pack", "a client pack position of at least " + std::to_string(last), jsonLine(pack));
				last = pack.Integer();
				observations.emplace(static_cast<uint64_t>(last), record);
				++loaded;
			}
			observationsLoaded[player] = loaded;
			observationsChecked[player] = 0;
		}
	}

	void openTrace(const std::string & path)
	{
		trace.open(path, std::ios::out | std::ios::trunc);
		if(!trace)
			throw std::runtime_error("Cannot write " + path);
	}

#ifdef HEROES_CONTEXTUAL_REPLAY
	void setContextual(ContextualReplay & value) { contextual = &value; }
#endif
	JsonNode run(const std::vector<JsonNode> & records, bool allowPlatformMismatch, bool prefix, const std::string & workDirectory)
	{
		if(records.size() < 2 || records[0]["kind"].String() != "init" || records[1]["kind"].String() != "start")
			throw std::runtime_error("Replay log must begin with init and start records");
		const auto & init = records[0];
		expectEqual("header", "schema", SCHEMA, init["schema"].String());
		expectEqual("header", "vcmi_version", GameConstants::VCMI_VERSION, init["vcmi_version"].String());
		expectEqual("header", "serialization_version", std::to_string(static_cast<int>(ESerializationVersion::CURRENT)), text(init["serialization_version"]));
		if(!allowPlatformMismatch)
			expectEqual("header", "platform", init["platform"].String(), platform());
		if(init["turn_timers"].Bool())
			throw std::runtime_error("Replays require games without turn timers");
		if(init["human_players"].isNull() || init["human_players"].Integer() >= 2)
			throw std::runtime_error("Replays require games with fewer than two human players");

		for(const auto & connection : init["connections"].Vector())
			for(const auto & player : connection["players"].Vector())
				server.connections[static_cast<int>(connection["id"].Integer())].insert(static_cast<int>(player.Integer()));
		for(const auto & host : init["hosts"].Vector())
			server.hosts.insert(static_cast<int>(host.Integer()));
		server.clientConnection = static_cast<int>(init["client_connection"].Integer());

		{
			Settings seed = settings.write["server"]["seed"];
			seed->Integer() = init["seed"].Integer();
		}
		// The game handler applies these exactly as in the recorded run. The
		// experiment directory only has to exist for its checks: the replay
		// records nothing, and the client code that writes evidence is absent.
		unsetenv("VCMI_EXPERIMENT_REPLAY");
		unsetenv("VCMI_RECORD_DIR");
		if(init["no_cheats"].isNull())
			unsetenv("VCMI_EXPERIMENT_NO_CHEATS");
		else
			setenv("VCMI_EXPERIMENT_NO_CHEATS", init["no_cheats"].String().c_str(), 1);
		setenv("VCMI_EXPERIMENT_DIR", workDirectory.c_str(), 1);

		StartInfo startInfo;
		decodeStartInfo(fromHex(init["start_info"].String()), startInfo);
		gameHandler = std::make_unique<CGameHandler>(server);
		server.gameHandler = gameHandler.get();
		Load::ProgressAccumulator progress;
		gameHandler->init(&startInfo, progress);
		expectState("init", init, *gameHandler);
#ifdef HEROES_CONTEXTUAL_REPLAY
		if(contextual)
		{
			contextual->initialize(*gameHandler->gs);
			server.clientPackProduced = [this](const CPackForClient & pack) { contextual->packet(pack); };
		}
#endif

		for(const auto & [connection, players] : server.connections)
			if(connection == server.clientConnection)
				for(const int player : players)
					if(player >= 0 && player < PlayerColor::PLAYER_LIMIT_I)
						callbacks[player] = std::make_shared<CCallback>(gameHandler->gs, PlayerColor(player), nullptr);
		server.clientPackApplied = [this](uint16_t type) { checkObservations(type); };
		writeTraceMap();

		gameHandler->start(false);
		server.setState(EServerState::GAMEPLAY);
		const auto & start = records[1];
		expectEqual("start", "out", text(start["out"]), std::to_string(server.outgoing.count));
		expectEqual("start", "client_out", text(start["client_out"]), std::to_string(server.outgoing.clientCount));
		expectChain("start", start, server.outgoing);
		expectState("start", start, *gameHandler);
		writeTrace(-1, nullptr);

		for(size_t index = 2; index < records.size(); index += 2)
		{
			const auto & input = records[index];
			if(input["kind"].String() != "input" || index + 1 >= records.size() || records[index + 1]["kind"].String() != "applied")
				throw std::runtime_error("Replay log must alternate input and applied records");
			const auto & applied = records[index + 1];
			const auto sequence = input["seq"].Integer();
			const std::string stage = "input " + std::to_string(sequence);
			expectEqual(stage, "seq", std::to_string(inputs), std::to_string(sequence));
			expectEqual(stage, "out_before", text(input["out"]), std::to_string(server.outgoing.count));

			std::unique_ptr<CPack> decoded;
			try
			{
				decoded = decodePack(fromHex(input["pack"].String()), &gameHandler->gameInfo());
			}
			catch(const std::runtime_error &)
			{
				throw Divergence(stage, "pack", "a complete recorded pack", "undecodable bytes");
			}
			auto * pack = dynamic_cast<CPackForServer *>(decoded.get());
			if(!pack)
				throw Divergence(stage, "pack", "a pack for the server", "another pack type");
			expectEqual(stage, "type", text(input["type"]), std::to_string(CTypeList::getInstance().getTypeID(pack)));
#ifdef HEROES_CONTEXTUAL_REPLAY
			if(contextual) contextual->input(input);
#endif

			server.outgoing.clearFlags();
			gameHandler->handleReceivedPack(GameConnectionID(static_cast<int8_t>(input["connection"].Integer())), *pack);
			++inputs;

			const char * boundary = boundaryAfter(*pack, gameHandler->lastPackResult, server.outgoing);
			const std::string after = "applied " + std::to_string(sequence);
			expectEqual(after, "blocked", text(applied["blocked"]), gameHandler->lastPackBlocked ? "true" : "false");
			expectEqual(after, "result", text(applied["result"]), gameHandler->lastPackResult ? "true" : "false");
			expectEqual(after, "out", text(applied["out"]), std::to_string(server.outgoing.count));
			expectEqual(after, "client_out", text(applied["client_out"]), std::to_string(server.outgoing.clientCount));
			expectChain(after, applied, server.outgoing);
			expectEqual(after, "boundary", text(applied["boundary"]), boundary ? boundary : "null");
			JsonNode replayedTerminals;
			replayedTerminals.Vector().clear();
			for(const auto & [player, victory] : server.outgoing.terminals)
			{
				JsonNode entry;
				entry["player"].Integer() = player;
				entry["victory"].Bool() = victory;
				replayedTerminals.Vector().push_back(entry);
			}
			expectEqual(after, "terminals", jsonLine(applied["terminals"]), jsonLine(replayedTerminals));
			if(boundary)
			{
				if(applied["digest"].isNull())
					throw Divergence(after, "digest", "null", digestHex(stateDigest(*gameHandler)));
				expectState(after, applied, *gameHandler);
				++boundaries[boundary];
				for(const auto & entry : replayedTerminals.Vector())
					terminals.Vector().push_back(entry);
			}
			writeTrace(sequence, boundary);
		}

		const auto unreached = observations.upper_bound(server.outgoing.clientCount);
		if(unreached != observations.end() && !prefix)
			throw Divergence("observation p" + std::to_string(unreached->second.player) + " " + unreached->second.kind + " "
				+ std::to_string(unreached->second.number), "pack", std::to_string(unreached->first),
				"replay ended at client pack " + std::to_string(server.outgoing.clientCount));
		std::map<int, int64_t> unchecked;
		for(auto entry = unreached; entry != observations.end(); ++entry)
			++unchecked[entry->second.player];
		for(const auto & [player, loaded] : observationsLoaded)
			expectEqual("observations p" + std::to_string(player), "records checked or after the prefix", std::to_string(loaded),
				std::to_string(observationsChecked[player] + unchecked[player]));

		JsonNode summary;
		summary["inputs"].Integer() = inputs;
		summary["prefix"].Bool() = prefix;
		summary["observations_after_prefix"].Integer() = std::distance(unreached, observations.end());
		summary["packs_sent"].Integer() = server.outgoing.count;
		summary["client_packs"].Integer() = server.outgoing.clientCount;
		for(const auto & [kind, count] : boundaries)
			summary["boundaries"][kind].Integer() = count;
		for(const auto & [player, count] : observationsChecked)
			summary["observations_checked"][std::to_string(player)].Integer() = count;
		for(const auto & [player, count] : observationsLoaded)
		{
			summary["observations_loaded"][std::to_string(player)].Integer() = count;
			summary["observations_unchecked"][std::to_string(player)].Integer() = unchecked[player];
		}
		terminals.Vector();
		summary["terminals"] = terminals;
		return summary;
	}

	JsonNode progress() const
	{
		JsonNode result;
		result["inputs"].Integer() = inputs;
		result["client_packs"].Integer() = server.outgoing.clientCount;
		for(const auto & [player, count] : observationsChecked)
			result["observations_checked"][std::to_string(player)].Integer() = count;
		for(const auto & [player, count] : observationsLoaded)
			result["observations_loaded"][std::to_string(player)].Integer() = count;
		return result;
	}
};

void writeReport(const std::string & path, const JsonNode & report)
{
	std::ofstream output(path, std::ios::out | std::ios::trunc);
	output << jsonLine(report) << '\n';
	if(!output)
		throw std::runtime_error("Cannot write " + path);
}
}

int main(int argc, const char * argv[])
{
	std::string logPath;
	std::string observationDirectory;
	std::string reportPath;
	std::string tracePath;
	std::string contextualPath;
	bool allowPlatformMismatch = false;
	bool prefix = false;
	for(int i = 1; i < argc; ++i)
	{
		const std::string argument = argv[i];
		auto value = [&]() -> std::string
		{
			if(i + 1 >= argc)
				throw std::runtime_error("Missing value for " + argument);
			return argv[++i];
		};
		if(argument == "--log")
			logPath = value();
		else if(argument == "--observations")
			observationDirectory = value();
		else if(argument == "--report")
			reportPath = value();
		else if(argument == "--trace")
			tracePath = value();
		else if(argument == "--contextual")
			contextualPath = value();
		else if(argument == "--allow-platform-mismatch")
			allowPlatformMismatch = true;
		else if(argument == "--prefix")
			prefix = true;
		else
		{
			std::cerr << "Unknown argument " << argument << "\n";
			return 2;
		}
	}
	if(logPath.empty() || reportPath.empty())
	{
		std::cerr << "Usage: vcmireplay --log replay.jsonl --report report.json [--observations dir] [--trace trace.jsonl] [--prefix]\n";
		return 2;
	}

	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_Replay_log.txt", &console);
	logConfigurator.configureDefault();
	JsonNode report;
	report["schema"].String() = contextualPath.empty() ? "heroes-replay-verification-v1" : "heroes-contextual-replay-verification-v1";
	report["log"].String() = logPath;
	report["verified"].Bool() = false;
	int status = 1;
	Replay replay;
	try
	{
		// Malformed evidence fails before the game library loads.
		const auto records = readLines(logPath);
		std::vector<JsonNode> contextualTape;
#ifdef HEROES_CONTEXTUAL_REPLAY
		ContextualReplay contextual;
		if(!contextualPath.empty())
		{
			if(allowPlatformMismatch || !observationDirectory.empty())
				throw std::runtime_error("Contextual replay requires exact platform and its own ledger");
			contextualTape = readLines(contextualPath);
			if(contextualTape.empty()) throw std::runtime_error("Missing contextual replay header");
			ContextualReplay::configureControllers(contextualTape.front());
			if(contextualTape.front()["schema"].String() == "heroes-contextual-builtin-replay-v1")
				report["schema"].String() = "heroes-contextual-builtin-replay-verification-v1";
			setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
			setenv("EXPLORER_RL_EPISODE", contextualTape.front()["episode"].String().c_str(), 1);
			setenv("EXPLORER_RL_INTERFACE", contextualTape.front()["interface"].String().c_str(), 1);
			replay.setContextual(contextual);
		}
		else unsetenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE");
#else
		if(!contextualPath.empty()) throw std::runtime_error("This server-only build has no contextual client replay");
#endif
		if(!observationDirectory.empty())
			replay.loadObservations(observationDirectory);
		LIBRARY = new GameLibrary;
		LIBRARY->initializeFilesystem(false);
		logConfigurator.configure();
		LIBRARY->initializeLibrary();
		if(!tracePath.empty())
			replay.openTrace(tracePath);
		report["summary"] = replay.run(records, allowPlatformMismatch, prefix,
			boost::filesystem::absolute(reportPath).parent_path().string());
#ifdef HEROES_CONTEXTUAL_REPLAY
		if(!contextualPath.empty())
		{
			auto serverSummary = report["summary"];
			report["summary"] = contextual.run(contextualTape, prefix);
			report["summary"]["server"] = serverSummary;
		}
#endif
		report["verified"].Bool() = true;
		report["divergence"] = JsonNode();
		status = 0;
	}
	catch(const Divergence & divergence)
	{
		report["divergence"] = divergence.details;
		report["summary"] = replay.progress();
	}
	catch(const std::exception & error)
	{
		report["error"].String() = error.what();
		report["summary"] = replay.progress();
		status = 3;
	}
	try
	{
		writeReport(reportPath, report);
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << "\n";
		status = 3;
	}
	std::cout << jsonLine(report) << "\n";
	logConfigurator.deconfigure();
	return status;
}
