#include "StdInc.h"
#include "ExplorerRL.h"
#include "ContextualEvidence.h"
#include "VisibleWorld.h"

#include "../../lib/battle/BattleAction.h"
#include "../../lib/battle/BattleSide.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/callback/ExperimentHooks.h"
#include "../../lib/gameState/CGameState.h"
#include "../../lib/gameState/EVictoryLossCheckResult.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"
#include "../../lib/mapping/TerrainTile.h"
#include "../../lib/networkPacks/PacksForClient.h"
#include "../../lib/networkPacks/PacksForClientBattle.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cerrno>
#include <cstdlib>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{
// Clockwise, beginning north. These are primitive one-tile commands.
constexpr std::array<std::array<int, 2>, 8> DIRECTIONS{{
	{{0, -1}}, {{1, -1}}, {{1, 0}}, {{1, 1}},
	{{0, 1}}, {{-1, 1}}, {{-1, 0}}, {{-1, -1}}
}};

int environmentInteger(const char * key, int fallback, int minimum, int maximum)
{
	const char * value = std::getenv(key);
	if(!value)
		return fallback;
	char * end = nullptr;
	const long result = std::strtol(value, &end, 10);
	if(!*value || *end || result < minimum || result > maximum)
		throw std::runtime_error(std::string("Invalid ") + key);
	return static_cast<int>(result);
}

int64_t microsecondsSince(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

JsonNode positionJson(const std::array<int, 3> & position)
{
	JsonNode result;
	for(const int coordinate : position)
		result.Vector().emplace_back(coordinate);
	return result;
}

JsonNode tileFactJson(const explorer_rl::TileFact & fact)
{
	JsonNode result;
	result["x"].Integer() = fact.position[0];
	result["y"].Integer() = fact.position[1];
	result["z"].Integer() = fact.position[2];
	result["terrain"].Integer() = fact.tile.terrain;
	result["blocked"].Bool() = fact.tile.blocked;
	result["visitable"].Bool() = fact.tile.visitable;
	return result;
}

JsonNode ownStateJson(const explorer_rl::OwnState & own)
{
	JsonNode result;
	result["hero"] = own.hero ? positionJson(*own.hero) : JsonNode();
	result["movement"].Integer() = own.movement;
	result["movement_max"].Integer() = own.movementMax;
	auto & resources = result["resources"].Vector();
	for(const auto amount : own.resources)
		resources.emplace_back(amount);
	auto & army = result["army"].Vector();
	army.clear();
	for(const auto & stack : own.army)
	{
		JsonNode entry;
		entry["slot"].Integer() = stack.slot;
		entry["creature"].Integer() = stack.creature;
		entry["count"].Integer() = stack.count;
		army.push_back(std::move(entry));
	}
	auto & towns = result["towns"].Vector();
	towns.clear();
	for(const auto & town : own.towns)
		towns.push_back(positionJson(town));
	result["mines"].Integer() = own.mines;
	return result;
}

JsonNode visibleObjectJson(const explorer_rl::VisibleObject & object)
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

JsonNode visibleChangeJson(const explorer_rl::VisibleChange & change, int64_t sequence, int day)
{
	JsonNode event;
	event["kind"].String() = "visible_update";
	event["sequence"].Integer() = sequence;
	event["day"].Integer() = day;
	auto & appeared = event["appeared"].Vector();
	for(const auto & object : change.appeared)
		appeared.push_back(visibleObjectJson(object));
	auto & updated = event["changed"].Vector();
	for(const auto & object : change.changed)
		updated.push_back(visibleObjectJson(object));
	auto & disappeared = event["disappeared"].Vector();
	for(const auto id : change.disappeared)
		disappeared.emplace_back(id);
	auto & revealed = event["revealed"].Vector();
	revealed.clear();
	for(const auto & fact : change.revealed)
		revealed.push_back(tileFactJson(fact));
	auto & tilesChanged = event["tiles_changed"].Vector();
	tilesChanged.clear();
	for(const auto & fact : change.tilesChanged)
		tilesChanged.push_back(tileFactJson(fact));
	auto & hidden = event["hidden"].Vector();
	hidden.clear();
	for(const auto & tile : change.hidden)
		hidden.push_back(positionJson(tile));
	event["own"] = change.own ? ownStateJson(*change.own) : JsonNode();
	return event;
}

void waitSocket(int fd, short events, int timeoutSeconds)
{
	pollfd descriptor{fd, events, 0};
	int result;
	do
		result = ::poll(&descriptor, 1, timeoutSeconds * 1000);
	while(result < 0 && errno == EINTR);
	if(result <= 0 || !(descriptor.revents & events))
		throw std::runtime_error(result == 0 ? "Policy socket timeout" : "Policy socket disconnected");
}

thread_local bool gameStateReadHeld = false;
}

ExplorerRL::GameStateRead::GameStateRead(const char * actor) : event(actor, "read")
{
	if(gameStateReadHeld)
		throw std::logic_error("Nested game-state read lock");
	lock = std::shared_lock(CGameState::mutex);
	gameStateReadHeld = true;
}

ExplorerRL::GameStateRead::~GameStateRead()
{
	gameStateReadHeld = false;
}

ExplorerRL::~ExplorerRL()
{
	finish();
}

void ExplorerRL::initGameInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CCallback> CB)
{
	env = std::move(ENV);
	callback = std::move(CB);
	cbc = callback;
	const auto callbackPlayer = callback->getPlayerID();
	if(!callbackPlayer)
		throw std::runtime_error("ExplorerRL requires a player-specific callback");
	playerID = *callbackPlayer;
	evidenceActor = "player" + std::to_string(playerID.getNum());
	if(const char * interface = std::getenv("EXPLORER_RL_INTERFACE"))
	{
		const std::string name(interface);
		if(name != "movement-v1" && name != "adventure-windows-v1" && name != "adventure-town-v1" && name != "adventure-armies-v1")
			throw std::runtime_error("Unsupported ExplorerRL interface");
		armyTransfer = name == "adventure-armies-v1";
		townEconomy = armyTransfer || name == "adventure-town-v1";
		contextual = townEconomy || name == "adventure-windows-v1";
	}
	if(contextual && playerID != PlayerColor(0) && playerID != PlayerColor(1))
		throw std::runtime_error("Contextual interface requires the two baseline players");
	callback->waitTillRealize = false;
	human = false;
	const char * episodeValue = std::getenv("EXPLORER_RL_EPISODE");
	episode = episodeValue ? episodeValue : "unspecified";
	if(contextual && contextualEvidenceEnabled())
		explorer_rl::ContextualEvidence::live(episode, townEconomy, armyTransfer);
	profile = environmentInteger("EXPLORER_RL_PROFILE", 0, 0, 1) == 1;
	timeoutSeconds = environmentInteger("EXPLORER_RL_TIMEOUT", 30, 1, 300);
	snapshotEvery = environmentInteger("EXPLORER_RL_SNAPSHOT_EVERY", 1, 0, 1000000);
	if(const char * replay = std::getenv("VCMI_EXPERIMENT_REPLAY"); replay && std::string(replay) == "1")
	{
		const char * directory = std::getenv("VCMI_EXPERIMENT_DIR");
		const std::string path = std::string(directory ? directory : "") + "/observations-p" + std::to_string(playerID.getNum()) + ".jsonl";
		observationLedger = directory && *directory ? std::fopen(path.c_str(), "w") : nullptr;
		if(observationLedger)
			ledgerEnabled = true;
		else
			failure = "Cannot create the replay observation ledger " + path; // reported by the worker
	}
	worker = std::thread([this] { run(); });
}

void ExplorerRL::finish()
{
	{
		std::lock_guard lock(stateMutex);
		stopping = true;
	}
	changed.notify_all();
	const int fd = socketFD.load();
	if(fd >= 0)
		::shutdown(fd, SHUT_RDWR);
	if(worker.joinable())
		worker.join();
	{
		std::lock_guard lock(stateMutex);
		if(observationLedger)
			std::fclose(observationLedger);
		observationLedger = nullptr;
	}
	const int closedFD = socketFD.exchange(-1);
	if(closedFD >= 0)
		::close(closedFD);
}

void ExplorerRL::connectPolicy()
{
	if(explorer_rl::ContextualEvidence::replaying())
		return;
	// Each instance has its own callback, state and socket. Plural routing never
	// falls back to a shared port, so a missing endpoint fails explicitly.
	const std::string portKey = std::getenv("EXPLORER_RL_PLAYERS")
		? "EXPLORER_RL_PORT_" + std::to_string(playerID.getNum()) : "EXPLORER_RL_PORT";
	const int port = environmentInteger(portKey.c_str(), 0, 1, 65535);
	if(port == 0)
		throw std::runtime_error(portKey + " is required");
	const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
	if(fd < 0)
		throw std::runtime_error("Cannot create policy socket");
	// Publish the descriptor only once connected, so that a terminal before
	// the connection fails explicitly instead of writing to a bare socket.
	struct CloseUnpublished
	{
		int fd;
		~CloseUnpublished() { if(fd >= 0) ::close(fd); }
	} closeUnpublished{fd};
#ifdef SO_NOSIGPIPE
	const int yes = 1;
	::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
	// Each message is one complete line; coalescing small writes only adds latency.
	const int noDelay = 1;
	::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
	timeval timeout{timeoutSeconds, 0};
	::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
	sockaddr_in address{};
	address.sin_family = AF_INET;
	address.sin_port = htons(static_cast<uint16_t>(port));
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
		throw std::runtime_error("Cannot connect to local policy server");
	closeUnpublished.fd = -1;
	{
		std::lock_guard lock(stateMutex);
		if(stopping)
		{
			::close(fd);
			throw std::runtime_error("Stopped while connecting to the policy server");
		}
		socketFD.store(fd);
	}
}

JsonNode ExplorerRL::message(const std::string & type) const
{
	JsonNode result;
	result["type"].String() = type;
	result["protocol_version"].Integer() = armyTransfer ? 6 : townEconomy ? 5 : contextual ? 4 : 3;
	result["player"].Integer() = playerID.getNum();
	result["episode"].String() = episode;
	return result;
}

bool ExplorerRL::buildAndSend(const std::function<JsonNode()> & build)
{
	ExperimentEventGuard event(evidenceActor.c_str(), "send");
	// gameOver writes the terminal under sendMutex on the network thread. A
	// message's events are taken and written under the same lock, so the
	// terminal either follows this message or this message is never built and
	// the terminal takes its events: the stream stays gap-free, terminal last.
	std::lock_guard lock(sendMutex);
	{
		std::lock_guard stateLock(stateMutex);
		if(terminal || stopping)
			return false;
	}
	writeMessage(build());
	return true;
}

void ExplorerRL::writeMessage(const JsonNode & message)
{
	if(auto * evidence = explorer_rl::ContextualEvidence::current())
	{
		evidence->message(playerID.getNum(), message);
		if(explorer_rl::ContextualEvidence::replaying())
			return;
	}
	const auto encodeStart = std::chrono::steady_clock::now();
	// VCMI's compact writer can retain formatting newlines. JSON strings escape
	// their own newlines, so removing literal CR/LF produces a JSONL record.
	std::string line = message.toCompactString();
	line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
	line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
	line += '\n';
	const auto encodeMicros = microsecondsSince(encodeStart);
	const int fd = socketFD.load();
	if(fd < 0)
		throw std::runtime_error("Policy socket is not connected");
	size_t offset = 0;
	while(offset < line.size())
	{
		waitSocket(fd, POLLOUT, timeoutSeconds);
#ifdef MSG_NOSIGNAL
		constexpr int sendFlags = MSG_NOSIGNAL;
#else
		constexpr int sendFlags = 0;
#endif
		const auto sent = ::send(fd, line.data() + offset, line.size() - offset, sendFlags);
		if(sent <= 0)
			throw std::runtime_error("Cannot write policy socket");
		offset += static_cast<size_t>(sent);
	}
	lastEncodeMicros = encodeMicros;
	lastSendMicros = microsecondsSince(encodeStart) - encodeMicros;
	lastMessageBytes = static_cast<int64_t>(line.size());
}

JsonNode ExplorerRL::receiveMessage()
{
	if(explorer_rl::ContextualEvidence::replaying())
		return explorer_rl::ContextualEvidence::current()->replayResponse(playerID.getNum());
	while(true)
	{
		const auto newline = receiveBuffer.find('\n');
		if(newline != std::string::npos)
		{
			if(newline > 65536)
				throw std::runtime_error("Policy response exceeds 64 KiB");
			std::string line = receiveBuffer.substr(0, newline);
			receiveBuffer.erase(0, newline + 1);
			JsonParsingSettings parsing;
			parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
			parsing.strict = true;
			JsonNode response(line.data(), line.size(), parsing, "ExplorerRL policy response");
			ExperimentEventGuard event(evidenceActor.c_str(), "receive");
			if(auto * evidence = explorer_rl::ContextualEvidence::current())
				evidence->response(playerID.getNum(), response);
			return response;
		}
		if(receiveBuffer.size() > 65536)
			throw std::runtime_error("Policy response exceeds 64 KiB");
		const int fd = socketFD.load();
		waitSocket(fd, POLLIN, timeoutSeconds);
		char buffer[4096];
		const auto received = ::recv(fd, buffer, sizeof(buffer), 0);
		if(received <= 0)
			throw std::runtime_error("Policy socket disconnected");
		receiveBuffer.append(buffer, static_cast<size_t>(received));
	}
}

bool ExplorerRL::ended()
{
	if(explorer_rl::ContextualEvidence::cancelled())
		return true;
	std::lock_guard lock(stateMutex);
	if(!failure.empty())
		throw std::runtime_error(failure);
	return stopping || terminal;
}

void ExplorerRL::run() noexcept
{
	try
	{
		connectPolicy();
		auto ready = message("ready");
		if(contextual)
		{
			ready["interface"].String() = armyTransfer ? "adventure-armies-v1" : townEconomy ? "adventure-town-v1" : "adventure-windows-v1";
			ready["history_schema"].String() = armyTransfer ? "heroes-player-history-v4" : townEconomy ? "heroes-player-history-v3" : "heroes-player-history-v2";
			ready["action_schema"].String() = armyTransfer ? "heroes-context-actions-v4" : townEconomy ? "heroes-context-actions-v3" : "heroes-context-actions-v2";
		}
		else
		{
			ready["adventure_actions"].Integer() = 9;
			ready["action_mask_semantics"].String() = "player-visible primitive attempts; server validates movement";
			ready["battle_ai"].String() = getBattleAIName();
			// Ordered player-visible changes between decisions; see VisibleState.h.
			ready["observation_events"].String() = "visible_update-v2";
			ready["profiling"].Bool() = profile;
			ready["snapshot_every"].Integer() = snapshotEvery;
			{
				const auto gameLock = readGameState();
				const auto size = callback->getMapSize();
				ready["map_size"] = positionJson({size.x, size.y, size.z}); // public map dimensions
			}
		}
		if(!buildAndSend([&ready] { return ready; }))
			return;
		while(true)
		{
			QueryID query;
			{
				std::unique_lock lock(stateMutex);
				changed.wait(lock, [this] { return stopping || terminal || !failure.empty() || !decisions.empty() || !turns.empty(); });
				if(stopping || terminal)
					return;
				if(!failure.empty())
					throw std::runtime_error(failure);
				if(!decisions.empty())
				{
					auto decision = std::move(decisions.front());
					decisions.pop_front();
					lock.unlock();
					handleDecision(std::move(decision));
					continue;
				}
				query = turns.front();
				turns.pop_front();
			}
			if(contextual)
				runContextualTurn(query);
			else
				runTurn(query);
		}
	}
	catch(const std::exception & error)
	{
		if(explorer_rl::ContextualEvidence::cancelled())
			return;
		bool report;
		{
			std::lock_guard lock(stateMutex);
			report = !stopping && !terminal;
			if(report)
				failure = error.what();
		}
		if(!report)
			return; // A simultaneous natural terminal must finish its socket write.
		logAi->error("ExplorerRL: %s", error.what());
		try
		{
			buildAndSend([this, &error]
			{
				if(contextual)
				{
					// Preserve the known failure prefix, including an actually sent
					// command without a result. Never invent completion or recapture
					// state after an unsupported interaction.
					auto pending = takeContextualRecords();
					if(!pending["records"].Vector().empty())
						writeMessage(pending);
				}
				auto result = message("error");
				result["message"].String() = error.what();
				return result;
			});
		}
		catch(...) {}
		// The error record is diagnostic; the client's failure exit is what
		// makes an unattended run end as a technical failure.
		reportExperimentFailure(std::string("ExplorerRL: ") + error.what());
		const int fd = socketFD.load();
		if(fd >= 0)
			::shutdown(fd, SHUT_RDWR);
	}
}

void ExplorerRL::yourTurn(QueryID queryID)
{
	{
		std::lock_guard lock(stateMutex);
		turns.push_back(queryID);
	}
	changed.notify_all();
}

void ExplorerRL::requestSent(const CPackForServer *, int requestID)
{
	if(std::this_thread::get_id() == worker.get_id())
	{
		std::unique_lock lock(stateMutex, std::defer_lock);
		if(!contextualIssueLockHeld)
			lock.lock();
		pendingRequest = requestID;
		if(contextual && contextState.issueArmed && contextState.executing && contextState.decision)
		{
			const auto command = contextState.commands++;
			contextState.requests.emplace(requestID, explorer_rl::ContextualState::Request{*contextState.decision, command});
			auto record = contextState.record("issued", callback->getDate(Date::DAY));
			record["decision_id"].Integer() = *contextState.decision;
			record["command_index"].Integer() = command;
			contextState.records.push_back(std::move(record));
			contextState.issueArmed = false;
		}
	}
}

void ExplorerRL::requestRealized(PackageApplied * pack)
{
	{
		std::lock_guard lock(stateMutex);
		// BattleAI uses the same callback. Parent moves and nested dialogs
		// have separate IDs; responses may precede requestSent returning.
		acknowledgements[pack->requestID] = pack->result;
		if(contextual)
		{
			const auto found = contextState.requests.find(pack->requestID);
			if(found != contextState.requests.end())
			{
				if(contextState.decision == found->second.decision && !terminal)
				{
					auto record = contextState.record("acknowledged", callback->getDate(Date::DAY));
					record["decision_id"].Integer() = found->second.decision;
					record["command_index"].Integer() = found->second.command;
					record["accepted"].Bool() = pack->result;
					contextState.records.push_back(std::move(record));
				}
				contextState.requests.erase(found);
			}
		}
	}
	changed.notify_all();
}

void ExplorerRL::playerBlocked(int reason, bool start)
{
	{
		std::lock_guard lock(stateMutex);
		if(reason == PlayerBlocked::ONGOING_MOVEMENT)
			moving = start;
		else if(reason == PlayerBlocked::UPCOMING_BATTLE && start)
			inBattle = true;
	}
	changed.notify_all();
}

void ExplorerRL::battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile,
	const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed)
{
	// The opponent kind is shown to a player entering its own battle.
	const bool left = side == BattleSide::LEFT_SIDE;
	const auto * opposingHero = left ? hero2 : hero1;
	const auto * opposingArmy = left ? army2 : army1;
	JsonNode event;
	event["kind"].String() = "battle_start";
	event["side"].String() = left ? "attacker" : "defender";
	// As SideInBattle::init assigns the neutral color: unowned armies and creature generators' guards.
	const auto * object = dynamic_cast<const CGObjectInstance *>(opposingArmy);
	const bool unowned = object && (!object->getOwner().isValidPlayer() || object->ID == Obj::CREATURE_GENERATOR1
		|| object->ID == Obj::CREATURE_GENERATOR2 || object->ID == Obj::CREATURE_GENERATOR3 || object->ID == Obj::CREATURE_GENERATOR4);
	event["opponent"].String() = explorer_rl::opponentKind(opposingHero != nullptr,
		dynamic_cast<const CGTownInstance *>(opposingArmy) != nullptr, unowned);
	if(contextual)
	{
		event.Struct().erase("opponent");
		contextualEvent(event);
	}
	{
		std::lock_guard lock(stateMutex);
		battleSide = side;
		battleWon.reset();
		if(!contextual)
			events.push_back(std::move(event));
	}
	if(!explorer_rl::ContextualEvidence::replaying())
		CAdventureAI::battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void ExplorerRL::battleEnd(const BattleID & battleID, const BattleResult * result, QueryID queryID)
{
	{
		std::lock_guard lock(stateMutex);
		if(result && battleSide && (result->winner == BattleSide::ATTACKER || result->winner == BattleSide::DEFENDER))
			battleWon = result->winner == *battleSide;
	}
	if(!explorer_rl::ContextualEvidence::replaying())
		CAdventureAI::battleEnd(battleID, result, queryID);
}

void ExplorerRL::battleEnded()
{
	{
		std::lock_guard lock(stateMutex);
		inBattle = false;
		++completedBattles;
		JsonNode event;
		event["kind"].String() = "battle_complete";
		if(battleWon)
			event["won"].Bool() = *battleWon;
		else
			event["won"] = JsonNode();
		battleSide.reset();
		battleWon.reset();
		if(contextual)
		{
			const auto day = callback->getDate(Date::DAY);
			if(!contextState.executing)
				contextState.supersede(day, "visible_event");
			++contextState.interactions;
			auto record = contextState.record("event", day);
			record["event"] = std::move(event);
			contextState.records.push_back(std::move(record));
		}
		else
			events.push_back(std::move(event));
	}
	changed.notify_all();
}

void ExplorerRL::heroVisit(const CGHeroInstance * hero, const CGObjectInstance * object, bool start)
{
	if(!start || !hero || !object || hero->getOwner() != playerID)
		return;
	if(contextual)
	{
		JsonNode event;
		event["kind"].String() = "visit";
		{
			std::lock_guard lock(stateMutex);
			event["subject"].String() = contextState.handle(object->id.getNum());
		}
		event["object_kind"].Integer() = object->ID.getNum();
		event["object_subtype"].Integer() = object->subID.getNum();
		event["owner"].String() = object->getOwner() == playerID ? "self" : object->getOwner().isValidPlayer() ? "opponent" : "neutral";
		contextualEvent(std::move(event));
		return;
	}
	// This event is explicitly delivered for our hero's own interaction.
	JsonNode event;
	event["kind"].String() = "visit";
	event["object_id"].Integer() = object->id.getNum();
	event["object_type"].Integer() = object->ID.getNum();
	event["object_subtype"].Integer() = object->subID.getNum();
	event["owner"].Integer() = object->getOwner().getNum();
	std::lock_guard lock(stateMutex);
	events.push_back(std::move(event));
}

JsonNode ExplorerRL::takeEvents()
{
	std::lock_guard lock(stateMutex);
	JsonNode result;
	result.Vector() = std::move(events);
	events.clear();
	return result;
}

ExplorerRL::GameStateRead ExplorerRL::readGameState() const
{
	return GameStateRead(evidenceActor.c_str());
}

void ExplorerRL::noteVisibleChanges()
{
	if(contextual)
	{
		captureContextual();
		return;
	}
	// Callers guarantee a stable state: the worker holds readGameState(), and
	// network callbacks run on the single thread that applies packets.
	const int day = callback->getDate(Date::DAY);
	// Before the first day nobody has acted, so nothing observable can be
	// missed; the first capture of day one reports the whole initial view.
	if(day < 1)
		return;
	const auto start = std::chrono::steady_clock::now();
	auto snapshot = explorer_rl::captureVisible(explorer_rl::CallbackWorld{*callback, playerID, callback->getMapSize()});
	const auto position = experimentPackPosition();
	const uint64_t digest = ledgerEnabled ? explorer_rl::snapshotDigest(snapshot) : 0;
	std::lock_guard lock(stateMutex);
	const auto change = visibleTracker.update(std::move(snapshot));
	++visibleCaptures;
	visibleCaptureMicros += microsecondsSince(start);
	lastCapture = Capture{position.index, position.type, digest};
	if(change.empty())
		return;
	if(observationLedger)
		std::fprintf(observationLedger, "{\"kind\":\"capture\",\"sequence\":%lld,\"pack\":%llu,\"pack_type\":%u,\"digest\":\"%016llx\"}\n",
			static_cast<long long>(visibleSequence), static_cast<unsigned long long>(position.index),
			static_cast<unsigned>(position.type), static_cast<unsigned long long>(digest));
	events.push_back(visibleChangeJson(change, visibleSequence++, day));
}

void ExplorerRL::noteLedgerMessage(const char * type)
{
	std::lock_guard lock(stateMutex);
	if(!observationLedger || !lastCapture)
		return;
	std::fprintf(observationLedger, "{\"kind\":\"message\",\"type\":\"%s\",\"index\":%lld,\"pack\":%llu,\"pack_type\":%u,\"digest\":\"%016llx\"}\n",
		type, static_cast<long long>(ledgerMessages++), static_cast<unsigned long long>(lastCapture->pack),
		static_cast<unsigned>(lastCapture->packType), static_cast<unsigned long long>(lastCapture->digest));
	// Before the message is sent: whatever a policy received has its engine
	// evidence on disk, even if the environment then stops the engine.
	std::fflush(observationLedger);
}

void ExplorerRL::visibleChangeTrigger() noexcept
{
	try
	{
		bool finalContextual = false;
		{
			std::lock_guard lock(stateMutex);
			if(stopping || (terminal && !contextual) || !failure.empty())
				return;
			finalContextual = contextual && terminal;
		}
		if(finalContextual)
		{
			// Finalization belongs to the external process/stream verifier. Keep
			// later permitted final packs, without manufacturing another decision.
			std::lock_guard sendLock(sendMutex);
			captureContextual();
			auto envelope = takeContextualRecords();
			if(!envelope["records"].Vector().empty())
				writeMessage(envelope);
			return;
		}
		noteVisibleChanges();
	}
	catch(const std::exception & error)
	{
		bool finalFailure;
		{
			std::lock_guard lock(stateMutex);
			if(failure.empty())
				failure = std::string("Visible change capture failed: ") + error.what();
			finalFailure = terminal;
		}
		changed.notify_all();
		if(finalFailure)
			reportExperimentFailure("ExplorerRL final visible capture failed");
	}
}

JsonNode ExplorerRL::takeProfile()
{
	JsonNode result;
	if(!profile)
		return result;
	result["previous_message_encode_us"].Integer() = lastEncodeMicros;
	result["previous_message_send_us"].Integer() = lastSendMicros;
	result["previous_message_bytes"].Integer() = lastMessageBytes;
	std::lock_guard lock(stateMutex);
	result["visible_captures"].Integer() = visibleCaptures;
	result["visible_capture_us"].Integer() = visibleCaptureMicros;
	visibleCaptures = 0;
	visibleCaptureMicros = 0;
	return result;
}

bool ExplorerRL::executeRequest(const std::function<void()> & request, bool waitForAdventure, int64_t policyActionID,
	bool * issued, const std::function<bool()> & beforeIssue)
{
	{
		std::lock_guard lock(stateMutex);
		pendingRequest = -1;
		if(waitForAdventure)
			acknowledgements.clear();
	}
	{
		const auto gameLock = readGameState();
		if(policyActionID >= 0)
		{
			// The consumption watermark is committed where no pack can be applied. The pack that ends this
			// player's game was either applied already, and the action is never issued, or is applied after
			// the request has left. gameOver runs only after that pack is applied and reads the watermark,
			// so it names exactly the actions issued before the end. Contextual
			// requests also serialize visible events with the actual native send.
			const bool playing = callback->getPlayerStatus(playerID, false) == EPlayerStatus::INGAME;
			std::unique_lock lock(stateMutex);
			if(!failure.empty())
				throw std::runtime_error(failure);
			if(!playing || stopping || terminal)
				return false;
			if(contextual)
			{
				if(contextState.decision != policyActionID)
					return false;
				if(contextState.routeInteractions && *contextState.routeInteractions != contextState.interactions)
				{
					contextState.issueArmed = false;
					return false;
				}
				if(beforeIssue && !beforeIssue())
					return false;
				submittedActionID = policyActionID;
				// Serialize the actual send and its requestSent notification with
				// visible events. A notice already queued cannot be followed by a
				// further route command. The synchronous notification uses this lock.
				contextualIssueLockHeld = true;
				try { request(); }
				catch(...) { contextualIssueLockHeld = false; throw; }
				contextualIssueLockHeld = false;
			}
			else
			{
				submittedActionID = policyActionID;
				lock.unlock();
				request();
			}
		}
		else
			request();
	}
	if(issued)
		*issued = true;
	std::unique_lock lock(stateMutex);
	// Keep this local: a nested dialog answer has its own request identifier.
	const int requestID = pendingRequest;
	if(requestID < 0)
		throw std::runtime_error("Engine did not send the requested action");
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
	while(true)
	{
		if(!failure.empty())
			throw std::runtime_error(failure);
		if(stopping || terminal)
			return false;
		// A battle can require a dialog response before movement can complete.
		// Never wait for parent movement/battle completion when answering it.
		if(!decisions.empty())
		{
			auto decision = std::move(decisions.front());
			decisions.pop_front();
			lock.unlock();
			handleDecision(std::move(decision));
			lock.lock();
			continue;
		}
		const auto acknowledged = acknowledgements.find(requestID);
		if(acknowledged != acknowledgements.end() && (!waitForAdventure || (!moving && !inBattle)))
		{
			const bool accepted = acknowledged->second;
			acknowledgements.erase(acknowledged);
			return accepted;
		}
		if(changed.wait_until(lock, deadline) == std::cv_status::timeout)
			throw std::runtime_error("Engine request or battle completion timed out");
	}
}

JsonNode ExplorerRL::playerState()
{
	// Callers hold readGameState() on the worker; see noteVisibleChanges().
	JsonNode result;
	result["day"].Integer() = callback->getDate(Date::DAY);
	const auto heroes = callback->getHeroesInfo();
	result["own_heroes"].Integer() = heroes.size();
	result["own_towns"].Integer() = callback->getTownsInfo().size();
	std::vector<int3> towns;
	for(const auto * town : callback->getTownsInfo())
		if(town->getOwner() == playerID)
			towns.push_back(town->visitablePos());
	std::sort(towns.begin(), towns.end(), [](const int3 & a, const int3 & b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); });
	auto & townPositions = result["towns"].Vector();
	townPositions.clear();
	for(const auto & town : towns)
		townPositions.push_back(positionJson({town.x, town.y, town.z}));
	int mines = 0;
	for(const auto * object : callback->getMyObjects())
		if(object->getOwner() == playerID && object->ID == Obj::MINE)
			++mines;
	result["own_mines"].Integer() = mines;
	for(int resource = 0; resource < GameConstants::RESOURCE_QUANTITY; ++resource)
		result["resources"].Vector().emplace_back(callback->getResourceAmount(GameResID(resource)));
	result["hero_id"].Integer() = -1;
	result["army"].Vector().clear();
	result["movement"].Integer() = 0;
	result["movement_max"].Integer() = 0;
	result["x"] = JsonNode();
	result["y"] = JsonNode();
	result["z"] = JsonNode();
	for(const auto * hero : heroes)
	{
		if(hero->getOwner() != playerID)
			continue;
		result["hero_id"].Integer() = hero->id.getNum();
		const auto position = hero->visitablePos();
		result["x"].Integer() = position.x;
		result["y"].Integer() = position.y;
		result["z"].Integer() = position.z;
		result["movement"].Integer() = hero->movementPointsRemaining();
		result["movement_max"].Integer() = hero->movementPointsLimit();
		for(const auto & [slot, stack] : hero->Slots())
		{
			JsonNode entry;
			entry["slot"].Integer() = slot.getNum();
			entry["creature"].Integer() = stack->getCreatureID().getNum();
			entry["count"].Integer() = stack->getCount();
			result["army"].Vector().push_back(std::move(entry));
		}
		break;
	}
	{
		// Every pack is captured as it is applied, so the latest capture
		// describes this state.
		std::lock_guard lock(stateMutex);
		result["visible_tiles"].Integer() = visibleTracker.visibleTiles();
		result["completed_battles"].Integer() = completedBattles;
	}
	return result;
}

JsonNode ExplorerRL::observation(std::array<bool, 9> & mask, ObjectInstanceID & heroID, int3 & position)
{
	// Callers hold readGameState(); see noteVisibleChanges().
	auto heroes = callback->getHeroesInfo();
	heroes.erase(std::remove_if(heroes.begin(), heroes.end(), [this](const auto * hero) { return hero->getOwner() != playerID; }), heroes.end());
	if(heroes.size() > 1)
		throw std::runtime_error("ExplorerRL scenario supports at most one owned hero");
	const auto * hero = heroes.empty() ? nullptr : heroes.front();
	heroID = hero ? hero->id : ObjectInstanceID(-1);
	position = hero ? hero->visitablePos() : int3(-1, -1, -1);
	const auto size = callback->getMapSize();
	auto result = message("observation");
	result["action_id"].Integer() = actionID;
	result["day"].Integer() = callback->getDate(Date::DAY);
	result["hero_id"].Integer() = heroID.getNum();
	if(hero)
	{
		result["x"].Integer() = position.x;
		result["y"].Integer() = position.y;
		result["z"].Integer() = position.z;
	}
	else
	{
		result["x"] = JsonNode();
		result["y"] = JsonNode();
		result["z"] = JsonNode();
	}
	result["width"].Integer() = size.x;
	result["height"].Integer() = size.y;
	result["levels"].Integer() = size.z;
	result["movement"].Integer() = hero ? hero->movementPointsRemaining() : 0;
	result["movement_max"].Integer() = hero ? hero->movementPointsLimit() : 0;
	// The tile list repeats what the events already established, so the
	// adapter can verify its fold. It may be sent periodically: events never are.
	const bool snapshot = snapshotEvery > 0 && observations++ % snapshotEvery == 0;
	auto & tiles = result["tiles"].Vector();
	for(int z = 0; snapshot && z < size.z; ++z)
		for(int y = 0; y < size.y; ++y)
			for(int x = 0; x < size.x; ++x)
			{
				const int3 location(x, y, z);
				if(!callback->isVisible(location))
					continue;
				const auto * tile = callback->getTile(location, false);
				if(!tile)
					continue;
				JsonNode item;
				item["x"].Integer() = x;
				item["y"].Integer() = y;
				item["z"].Integer() = z;
				const auto facts = explorer_rl::visibleTileFacts(*callback, *tile);
				item["terrain"].Integer() = facts.terrain;
				item["blocked"].Bool() = facts.blocked;
				item["visitable"].Bool() = facts.visitable;
				auto & objects = item["objects"].Vector();
				for(const auto objectID : tile->visitableObjects)
				{
					const auto * object = explorer_rl::playerVisibleObject(*callback, objectID);
					if(!object)
						continue;
					JsonNode entry;
					entry["id"].Integer() = object->id.getNum();
					entry["type"].Integer() = object->ID.getNum();
					entry["subtype"].Integer() = object->subID.getNum();
					entry["owner"].Integer() = object->getOwner().getNum();
					objects.push_back(std::move(entry));
				}
				tiles.push_back(std::move(item));
			}
	if(!snapshot)
		result.Struct().erase("tiles");
	mask.fill(false);
	mask[8] = true;
	const auto * source = hero ? callback->getTile(position, false) : nullptr;
	for(size_t i = 0; i < DIRECTIONS.size(); ++i)
	{
		const int3 destination(position.x + DIRECTIONS[i][0], position.y + DIRECTIONS[i][1], position.z);
		if(!hero || hero->movementPointsRemaining() <= 0 || !callback->isInTheMap(destination))
			continue;
		// Never query a hidden tile to construct an action mask. An unobserved
		// neighbor remains a possible attempt and the server judges validity.
		if(!callback->isVisible(destination))
		{
			mask[i] = true;
			continue;
		}
		const auto * target = callback->getTile(destination, false);
		if(!target || !source)
			continue;
		const auto facts = explorer_rl::visibleTileFacts(*callback, *target);
		mask[i] = target->entrableTerrain(source) && (!facts.blocked || facts.visitable);
	}
	for(bool allowed : mask)
		result["action_mask"].Vector().emplace_back(allowed);
	return result;
}

void ExplorerRL::runTurn(QueryID queryID)
{
	// VCMI uses -1 for a normal initial turn without a pending query.
	// Like its built-in AI, acknowledge only a real query identifier.
	if(queryID != QueryID(-1) && !executeRequest([this, queryID] { callback->selectionMade(0, queryID); }))
	{
		if(!ended())
			throw std::runtime_error("Engine rejected turn acknowledgement");
		return;
	}
	while(!ended())
	{
		std::array<bool, 9> mask;
		ObjectInstanceID heroID;
		int3 position;
		const bool sent = buildAndSend([&]
		{
			const auto buildStart = std::chrono::steady_clock::now();
			JsonNode state;
			{
				// No packet can be applied while this lock is held, so the events
				// end exactly at the state described by this decision snapshot.
				// A pack applied just before it may not have reached packApplied
				// yet, so this capture is required, not redundant.
				const auto gameLock = readGameState();
				noteVisibleChanges();
				state = observation(mask, heroID, position);
				state["state"] = playerState();
				state["events"] = takeEvents();
				noteLedgerMessage("observation");
			}
			if(profile)
			{
				state["perf"] = takeProfile();
				state["perf"]["build_us"].Integer() = microsecondsSince(buildStart);
			}
			return state;
		});
		if(!sent)
			return;
		const auto response = receiveMessage();
		if(ended())
			return;
		if(response["action_id"].getType() != JsonNode::JsonType::DATA_INTEGER || response["action_id"].Integer() != actionID || response["action"].getType() != JsonNode::JsonType::DATA_INTEGER)
			throw std::runtime_error("Policy response must contain matching action_id and integer action");
		const auto action = response["action"].Integer();
		if(action < 0 || action >= static_cast<int64_t>(mask.size()) || !mask[action])
			throw std::runtime_error("Policy selected a masked or unsupported action");
		JsonNode before;
		{
			const auto gameLock = readGameState();
			before = playerState();
		}
		const auto requestStart = std::chrono::steady_clock::now();
		bool issued = false;
		const bool accepted = executeRequest([this, action, heroID, position] {
			if(action == 8)
			{
				callback->endTurn();
				return;
			}
			const auto * hero = callback->getHero(heroID);
			if(!hero || hero->getOwner() != playerID || hero->visitablePos() != position)
				throw std::runtime_error("Hero changed before its primitive action");
			const int3 destination(position.x + DIRECTIONS[action][0], position.y + DIRECTIONS[action][1], position.z);
			callback->moveHero(hero, hero->convertFromVisitablePos(destination), false);
		}, true, actionID, &issued);
		// An action the game ended before is never issued; the terminal's watermark then excludes it.
		if(!issued || ended())
			return;
		const auto requestMicros = microsecondsSince(requestStart);
		const bool resultSent = buildAndSend([&]
		{
			auto result = message("action_result");
			result["action_id"].Integer() = actionID;
			result["action"].Integer() = action;
			result["accepted"].Bool() = accepted;
			result["before"] = before;
			{
				const auto gameLock = readGameState();
				noteVisibleChanges();
				result["after"] = playerState();
				result["events"] = takeEvents();
				noteLedgerMessage("action_result");
			}
			if(profile)
			{
				result["perf"] = takeProfile();
				result["perf"]["request_us"].Integer() = requestMicros;
			}
			return result;
		});
		if(!resultSent)
			return;
		++actionID;
		if(action == 8 && accepted)
		{
			buildAndSend([this] { return message("turn_end"); });
			return;
		}
	}
}

void ExplorerRL::gameOver(PlayerColor player, const EVictoryLossCheckResult & result)
{
	if(player != playerID || (!result.victory() && !result.loss()))
		return;
	std::string earlierFailure;
	{
		std::lock_guard lock(stateMutex);
		if(terminal)
			return;
		terminal = true;
		earlierFailure = failure;
	}
	changed.notify_all();
	if(!earlierFailure.empty())
	{
		// A failed episode never becomes a natural ending, even when the game
		// reaches one before the worker has reported the failure.
		reportExperimentFailure("ExplorerRL failed before the game ended: " + earlierFailure);
		return;
	}
	try
	{
		// Holding sendMutex first lets an in-flight worker message finish, or
		// return its events, before this record takes the remaining ones.
		std::lock_guard sendLock(sendMutex);
		if(contextual)
		{
			contextualTerminal(result.victory() ? "win" : "loss");
			return;
		}
		auto record = message("terminal");
		record["outcome"].String() = result.victory() ? "win" : "loss";
		{
			// The last action issued before this player's game ended (see executeRequest); a later reply never is.
			std::lock_guard lock(stateMutex);
			record["last_action_id"].Integer() = submittedActionID;
		}
		// Packet callbacks run on the applying thread, so the state is stable here.
		noteVisibleChanges();
		record["events"] = takeEvents();
		record["state"] = playerState();
		noteLedgerMessage("terminal");
		writeMessage(record);
		std::lock_guard lock(stateMutex);
		if(observationLedger)
			std::fflush(observationLedger);
	}
	catch(const std::exception & error)
	{
		logAi->error("ExplorerRL terminal delivery failed: %s", error.what());
		reportExperimentFailure(std::string("ExplorerRL terminal delivery failed: ") + error.what());
	}
}

void ExplorerRL::unsupported(const std::string & operation)
{
	bool finalFailure;
	{
		std::lock_guard lock(stateMutex);
		failure = "Unsupported scenario interaction: " + operation;
		finalFailure = terminal;
	}
	changed.notify_all();
	if(finalFailure)
		reportExperimentFailure("ExplorerRL unsupported final interaction: " + operation);
}

void ExplorerRL::enqueueDecision(QueryID query, JsonNode payload)
{
	if(contextual)
	{
		unsupported("unsupported_interaction: query dialog");
		return;
	}
	if(query == QueryID(-1))
	{
		unsupported("non-query dialog");
		return;
	}
	{
		std::lock_guard lock(stateMutex);
		decisions.push_back({query, std::move(payload)});
	}
	changed.notify_all();
}

void ExplorerRL::handleDecision(Decision decision)
{
	auto request = message("decision");
	for(const auto & [key, value] : decision.payload.Struct())
		request[key] = value;
	const auto id = decisionID++;
	request["decision_id"].Integer() = id;
	const bool sent = buildAndSend([&]
	{
		auto built = request;
		const auto gameLock = readGameState();
		noteVisibleChanges();
		built["events"] = takeEvents();
		noteLedgerMessage("decision");
		return built;
	});
	if(!sent)
		return;
	const auto response = receiveMessage();
	if(ended())
		return;
	if(response["decision_id"].getType() != JsonNode::JsonType::DATA_INTEGER || response["decision_id"].Integer() != id || response["choice"].getType() != JsonNode::JsonType::DATA_INTEGER)
		throw std::runtime_error("Policy decision requires matching decision_id and integer choice");
	const auto choice = response["choice"].Integer();
	bool allowed = false;
	for(const auto & entry : request["choices"].Vector())
		allowed |= entry["choice"].Integer() == choice;
	if(!allowed)
		throw std::runtime_error("Policy selected an unsupported dialog choice");
	const bool accepted = executeRequest([this, choice, query = decision.query] {
		callback->selectionMade(static_cast<int>(choice), query);
	}, false);
	if(!accepted && !ended())
		throw std::runtime_error("Engine rejected the policy's dialog choice");
	JsonNode event;
	event["kind"].String() = "decision";
	event["decision_kind"] = request["kind"];
	event["decision_id"].Integer() = id;
	event["choice"].Integer() = choice;
	std::lock_guard lock(stateMutex);
	events.push_back(std::move(event));
}

void ExplorerRL::heroGotLevel(const CGHeroInstance * hero, PrimarySkill primary, std::vector<SecondarySkill> & skills, QueryID query)
{
	if(contextual)
	{
		unsupported("unsupported_interaction: hero level-up");
		return;
	}
	if(!hero || hero->getOwner() != playerID)
	{
		unsupported("level-up of a hero the player does not own");
		return;
	}
	if(query == QueryID(-1))
	{
		// With at most one secondary skill offered the server applies the
		// level-up itself and opens no query: there is nothing to choose.
		JsonNode event;
		event["kind"].String() = "level_up";
		event["primary_skill"].Integer() = primary.getNum();
		auto & offered = event["secondary_skills"].Vector();
		offered.clear();
		for(const auto & skill : skills)
			offered.emplace_back(skill.getNum());
		std::lock_guard lock(stateMutex);
		events.push_back(std::move(event));
		return;
	}
	if(skills.empty())
	{
		unsupported("level-up query without choices");
		return;
	}
	JsonNode payload;
	payload["kind"].String() = "hero_level_up";
	payload["hero_id"].Integer() = hero->id.getNum();
	payload["primary_skill"].Integer() = primary.getNum();
	for(size_t i = 0; i < skills.size(); ++i)
	{
		JsonNode choice;
		choice["choice"].Integer() = i;
		choice["secondary_skill"].Integer() = skills[i].getNum();
		payload["choices"].Vector().push_back(std::move(choice));
	}
	enqueueDecision(query, std::move(payload));
}

void ExplorerRL::showBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID query, int, bool selection, bool cancel, bool safeToAutoaccept)
{
	if(contextual)
	{
		unsupported("unsupported_interaction: blocking dialog");
		return;
	}
	// These flags describe VCMI's ordinary finite-choice query protocol.
	// Unknown window-specific operations are handled separately and rejected.
	if(selection && components.empty())
	{
		unsupported("selection dialog without choices");
		return;
	}
	JsonNode payload;
	payload["kind"].String() = selection ? "selection" : cancel ? "yes_no" : "acknowledge";
	payload["text"].String() = text;
	payload["safe_to_autoaccept"].Bool() = safeToAutoaccept;
	const int first = selection && !cancel ? 1 : 0;
	const int last = selection ? static_cast<int>(components.size()) : cancel ? 1 : 0;
	for(int i = first; i <= last; ++i)
	{
		JsonNode choice;
		choice["choice"].Integer() = i;
		if(selection && i > 0)
		{
			const auto & component = components.at(i - 1);
			choice["component_type"].Integer() = static_cast<int>(component.type);
			choice["component_subtype"].Integer() = component.subType.getNum();
			if(component.value)
				choice["value"].Integer() = *component.value;
		}
		payload["choices"].Vector().push_back(std::move(choice));
	}
	enqueueDecision(query, std::move(payload));
}

void ExplorerRL::showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID query, const MetaString &)
{
	// The first action interface has no army-transfer action. Closing this
	// window is still an explicit, logged external choice, never a strategy.
	JsonNode payload;
	payload["kind"].String() = "garrison_close_only";
	JsonNode choice;
	choice["choice"].Integer() = 0;
	payload["choices"].Vector().push_back(std::move(choice));
	enqueueDecision(query, std::move(payload));
}

void ExplorerRL::commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) { unsupported("commander level-up"); }
void ExplorerRL::showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) { unsupported("teleport"); }
void ExplorerRL::showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) { unsupported("object selection"); }
void ExplorerRL::showRecruitmentDialog(const CGDwelling *, const CArmedInstance *, int, QueryID) { unsupported("dwelling recruitment"); }
void ExplorerRL::showMarketWindow(const IMarket *, const CGHeroInstance *, QueryID) { unsupported("market"); }
void ExplorerRL::showUniversityWindow(const IMarket *, const CGHeroInstance *, QueryID) { unsupported("university"); }
void ExplorerRL::showTavernWindow(const CGObjectInstance *, const CGHeroInstance *, QueryID) { unsupported("tavern"); }
void ExplorerRL::heroExchangeStarted(ObjectInstanceID, ObjectInstanceID, QueryID) { unsupported("hero exchange"); }

void ExplorerRL::packApplied() { visibleChangeTrigger(); }

std::optional<BattleAction> ExplorerRL::makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &)
{
	// Auto-battle is an explicitly fixed part of this exploration experiment.
	// No adventure-layer tactical surrender policy is supplied.
	return std::nullopt;
}
