#include "StdInc.h"
#include "ExplorerRL.h"

#include "../../lib/battle/BattleAction.h"
#include "../../lib/battle/BattleSide.h"
#include "../../lib/callback/CCallback.h"
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

/// Terrain and flags of a tile the player sees, from the objects it can see there.
explorer_rl::VisibleTile visibleTileFacts(const CCallback & callback, const TerrainTile & tile)
{
	auto anyVisible = [&callback](const std::vector<ObjectInstanceID> & objects)
	{
		return std::any_of(objects.begin(), objects.end(), [&callback](const auto objectID)
		{
			const auto * object = callback.getObj(objectID, false);
			return object && callback.isVisible(object);
		});
	};
	return {tile.getTerrainID().getNum(), anyVisible(tile.blockingObjects), anyVisible(tile.visitableObjects)};
}

/// The player's own callback as a VisibleState world. Objects are listed only
/// for tiles its team sees, and only when the callback reports them visible.
struct CallbackWorld
{
	const CCallback & callback;
	PlayerColor player;
	int3 size;

	int width() const { return size.x; }
	int height() const { return size.y; }
	int levels() const { return size.z; }
	bool tileVisible(int x, int y, int z) const { return callback.isVisible(int3(x, y, z)); }

	explorer_rl::VisibleTile tileFacts(int x, int y, int z) const
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile)
			throw std::runtime_error("A visible tile has no terrain");
		return visibleTileFacts(callback, *tile);
	}

	void visibleObjectsAt(int x, int y, int z, std::vector<explorer_rl::VisibleObject> & out) const
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile)
			return;
		for(const auto objectID : tile->visitableObjects)
		{
			const auto * object = callback.getObj(objectID, false);
			if(!object || !callback.isVisible(object))
				continue;
			out.push_back({object->id.getNum(), x, y, z, object->ID.getNum(), object->subID.getNum(), object->getOwner().getNum()});
		}
	}

	explorer_rl::OwnState ownState() const
	{
		explorer_rl::OwnState own;
		for(int resource = 0; resource < GameConstants::RESOURCE_QUANTITY; ++resource)
			own.resources.push_back(callback.getResourceAmount(GameResID(resource)));
		for(const auto * hero : callback.getHeroesInfo())
		{
			if(hero->getOwner() != player)
				continue;
			const auto position = hero->visitablePos();
			own.hero = std::array<int, 3>{position.x, position.y, position.z};
			own.movement = hero->movementPointsRemaining();
			own.movementMax = hero->movementPointsLimit();
			for(const auto & [slot, stack] : hero->Slots())
				own.army.push_back({slot.getNum(), stack->getCreatureID().getNum(), stack->getCount()});
			break;
		}
		for(const auto * town : callback.getTownsInfo())
			if(town->getOwner() == player)
			{
				const auto position = town->visitablePos();
				own.towns.push_back({position.x, position.y, position.z});
			}
		std::sort(own.towns.begin(), own.towns.end());
		for(const auto * object : callback.getMyObjects())
			if(object->getOwner() == player && object->ID == Obj::MINE)
				++own.mines;
		return own;
	}
};

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
	playerID = *callback->getPlayerID();
	callback->waitTillRealize = false;
	human = false;
	const char * episodeValue = std::getenv("EXPLORER_RL_EPISODE");
	episode = episodeValue ? episodeValue : "unspecified";
	profile = environmentInteger("EXPLORER_RL_PROFILE", 0, 0, 1) == 1;
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
	const int closedFD = socketFD.exchange(-1);
	if(closedFD >= 0)
		::close(closedFD);
}

void ExplorerRL::connectPolicy()
{
	// Each instance has its own callback, state and socket. Plural routing never
	// falls back to a shared port, so a missing endpoint fails explicitly.
	const std::string portKey = std::getenv("EXPLORER_RL_PLAYERS")
		? "EXPLORER_RL_PORT_" + std::to_string(playerID.getNum()) : "EXPLORER_RL_PORT";
	const int port = environmentInteger(portKey.c_str(), 0, 1, 65535);
	if(port == 0)
		throw std::runtime_error(portKey + " is required");
	timeoutSeconds = environmentInteger("EXPLORER_RL_TIMEOUT", 30, 1, 300);
	const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
	if(fd < 0)
		throw std::runtime_error("Cannot create policy socket");
	socketFD.store(fd);
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
}

JsonNode ExplorerRL::message(const std::string & type) const
{
	JsonNode result;
	result["type"].String() = type;
	result["protocol_version"].Integer() = 3;
	result["player"].Integer() = playerID.getNum();
	result["episode"].String() = episode;
	return result;
}

void ExplorerRL::sendMessage(const JsonNode & message)
{
	std::lock_guard lock(sendMutex);
	{
		std::lock_guard stateLock(stateMutex);
		// gameOver runs on the network thread. Either this write completes
		// before its terminal write, or it is suppressed: terminal stays last.
		if(terminal)
		{
			// The terminal record is assembled under sendMutex after this
			// returns, so returning these events keeps the history gap-free.
			const auto & undelivered = message["events"];
			if(undelivered.getType() == JsonNode::JsonType::DATA_VECTOR)
				events.insert(events.begin(), undelivered.Vector().begin(), undelivered.Vector().end());
			return;
		}
	}
	writeMessage(message);
}

void ExplorerRL::writeMessage(const JsonNode & message)
{
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
			return JsonNode(line.data(), line.size(), parsing, "ExplorerRL policy response");
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
		ready["adventure_actions"].Integer() = 9;
		ready["action_mask_semantics"].String() = "player-visible primitive attempts; server validates movement";
		ready["battle_ai"].String() = getBattleAIName();
		// Ordered player-visible changes between decisions; see VisibleState.h.
		ready["observation_events"].String() = "visible_update-v2";
		ready["profiling"].Bool() = profile;
		{
			auto gameState = readGameState();
			const auto size = callback->getMapSize();
			ready["map_size"] = positionJson({size.x, size.y, size.z}); // public map dimensions
		}
		sendMessage(ready);
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
			runTurn(query);
		}
	}
	catch(const std::exception & error)
	{
		bool report;
		{
			std::lock_guard lock(stateMutex);
			report = !stopping && !terminal;
			if(report)
				failure = error.what();
		}
		if(!report)
			return; // A simultaneous natural terminal must finish its socket write.
		if(report)
		{
			logAi->error("ExplorerRL: %s", error.what());
			try
			{
				auto result = message("error");
				result["message"].String() = error.what();
				sendMessage(result);
			}
			catch(...) {}
		}
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
		std::lock_guard lock(stateMutex);
		pendingRequest = requestID;
	}
}

void ExplorerRL::requestRealized(PackageApplied * pack)
{
	{
		std::lock_guard lock(stateMutex);
		// BattleAI uses the same callback. Parent moves and nested dialogs
		// have separate IDs; responses may precede requestSent returning.
		acknowledgements[pack->requestID] = pack->result;
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
	if(opposingHero)
		event["opponent"].String() = "hero";
	else if(dynamic_cast<const CGTownInstance *>(opposingArmy))
		event["opponent"].String() = "town_garrison";
	else if(const auto * object = dynamic_cast<const CGObjectInstance *>(opposingArmy); object && object->getOwner() == PlayerColor::NEUTRAL)
		event["opponent"].String() = "neutral";
	else
		event["opponent"].String() = "other";
	{
		std::lock_guard lock(stateMutex);
		battleSide = side;
		battleWon.reset();
		events.push_back(std::move(event));
	}
	CAdventureAI::battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void ExplorerRL::battleEnd(const BattleID & battleID, const BattleResult * result, QueryID queryID)
{
	{
		std::lock_guard lock(stateMutex);
		if(result && battleSide && (result->winner == BattleSide::ATTACKER || result->winner == BattleSide::DEFENDER))
			battleWon = result->winner == *battleSide;
	}
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
		events.push_back(std::move(event));
	}
	changed.notify_all();
}

void ExplorerRL::heroVisit(const CGHeroInstance * hero, const CGObjectInstance * object, bool start)
{
	if(!start || !hero || !object || hero->getOwner() != playerID)
		return;
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

std::shared_lock<std::shared_mutex> ExplorerRL::readGameState() const
{
	return std::shared_lock(CGameState::mutex);
}

void ExplorerRL::noteVisibleChanges()
{
	// Callers guarantee a stable state: the worker holds readGameState(), and
	// network callbacks run on the single thread that applies packets.
	const int day = callback->getDate(Date::DAY);
	// Before the first day nobody has acted, so nothing observable can be
	// missed; the first capture of day one reports the whole initial view.
	if(day < 1)
		return;
	const auto start = std::chrono::steady_clock::now();
	auto snapshot = explorer_rl::captureVisible(CallbackWorld{*callback, playerID, callback->getMapSize()});
	std::lock_guard lock(stateMutex);
	const auto change = visibleTracker.update(std::move(snapshot));
	++visibleCaptures;
	visibleCaptureMicros += microsecondsSince(start);
	if(!change.empty())
		events.push_back(visibleChangeJson(change, visibleSequence++, day));
}

void ExplorerRL::visibleChangeTrigger() noexcept
{
	try
	{
		{
			std::lock_guard lock(stateMutex);
			if(stopping || terminal || !failure.empty())
				return;
		}
		noteVisibleChanges();
	}
	catch(const std::exception & error)
	{
		{
			std::lock_guard lock(stateMutex);
			if(failure.empty())
				failure = std::string("Visible change capture failed: ") + error.what();
		}
		changed.notify_all();
	}
}

JsonNode ExplorerRL::takeProfile()
{
	JsonNode result;
	if(!profile)
		return result;
	{
		std::lock_guard lock(sendMutex);
		result["previous_message_encode_us"].Integer() = lastEncodeMicros;
		result["previous_message_send_us"].Integer() = lastSendMicros;
		result["previous_message_bytes"].Integer() = lastMessageBytes;
	}
	std::lock_guard lock(stateMutex);
	result["visible_captures"].Integer() = visibleCaptures;
	result["visible_capture_us"].Integer() = visibleCaptureMicros;
	visibleCaptures = 0;
	visibleCaptureMicros = 0;
	return result;
}

bool ExplorerRL::executeRequest(const std::function<void()> & request, bool waitForAdventure)
{
	{
		std::lock_guard lock(stateMutex);
		pendingRequest = -1;
		if(waitForAdventure)
			acknowledgements.clear();
	}
	{
		std::shared_lock gameLock(CGameState::mutex);
		request();
	}
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
	const auto size = callback->getMapSize();
	int visibleTiles = 0;
	for(int z = 0; z < size.z; ++z)
		for(int y = 0; y < size.y; ++y)
			for(int x = 0; x < size.x; ++x)
				visibleTiles += callback->isVisible(int3(x, y, z)) ? 1 : 0;
	result["visible_tiles"].Integer() = visibleTiles;
	{
		std::lock_guard lock(stateMutex);
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
	auto & tiles = result["tiles"].Vector();
	for(int z = 0; z < size.z; ++z)
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
				const auto facts = visibleTileFacts(*callback, *tile);
				item["terrain"].Integer() = facts.terrain;
				item["blocked"].Bool() = facts.blocked;
				item["visitable"].Bool() = facts.visitable;
				auto & objects = item["objects"].Vector();
				for(const auto objectID : tile->visitableObjects)
				{
					const auto * object = callback->getObj(objectID, false);
					if(!object || !callback->isVisible(object))
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
		mask[i] = target && source && target->entrableTerrain(source) && (!target->blocked() || target->visitable());
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
		JsonNode state;
		const auto buildStart = std::chrono::steady_clock::now();
		{
			// No packet can be applied while this lock is held, so the events end
			// exactly at the state described by this decision snapshot.
			const auto gameLock = readGameState();
			noteVisibleChanges();
			state = observation(mask, heroID, position);
			state["state"] = playerState();
			state["events"] = takeEvents();
		}
		if(profile)
		{
			state["perf"] = takeProfile();
			state["perf"]["build_us"].Integer() = microsecondsSince(buildStart);
		}
		sendMessage(state);
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
		});
		if(ended())
			return;
		const auto requestMicros = microsecondsSince(requestStart);
		auto result = message("action_result");
		result["action_id"].Integer() = actionID++;
		result["action"].Integer() = action;
		result["accepted"].Bool() = accepted;
		result["before"] = before;
		{
			const auto gameLock = readGameState();
			noteVisibleChanges();
			result["after"] = playerState();
			result["events"] = takeEvents();
		}
		if(profile)
		{
			result["perf"] = takeProfile();
			result["perf"]["request_us"].Integer() = requestMicros;
		}
		sendMessage(result);
		if(action == 8 && accepted)
		{
			sendMessage(message("turn_end"));
			return;
		}
	}
}

void ExplorerRL::gameOver(PlayerColor player, const EVictoryLossCheckResult & result)
{
	if(player != playerID || (!result.victory() && !result.loss()))
		return;
	{
		std::lock_guard lock(stateMutex);
		if(terminal)
			return;
		terminal = true;
	}
	changed.notify_all();
	try
	{
		// Holding sendMutex first lets an in-flight worker message finish, or
		// return its events, before this record takes the remaining ones.
		std::lock_guard sendLock(sendMutex);
		auto record = message("terminal");
		record["outcome"].String() = result.victory() ? "win" : "loss";
		// Packet callbacks run on the applying thread, so the state is stable here.
		noteVisibleChanges();
		record["events"] = takeEvents();
		record["state"] = playerState();
		writeMessage(record);
	}
	catch(const std::exception & error)
	{
		logAi->error("ExplorerRL terminal delivery failed: %s", error.what());
	}
}

void ExplorerRL::unsupported(const std::string & operation)
{
	{
		std::lock_guard lock(stateMutex);
		failure = "Unsupported scenario interaction: " + operation;
	}
	changed.notify_all();
}

void ExplorerRL::enqueueDecision(QueryID query, JsonNode payload)
{
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
	{
		const auto gameLock = readGameState();
		noteVisibleChanges();
		request["events"] = takeEvents();
	}
	sendMessage(request);
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
	if(!hero || hero->getOwner() != playerID || skills.empty())
	{
		unsupported("level-up without owned hero or choices");
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

void ExplorerRL::heroMoved(const TryMoveHero &, bool) { visibleChangeTrigger(); }
void ExplorerRL::objectRemovedAfter() { visibleChangeTrigger(); }
void ExplorerRL::newObject(const CGObjectInstance *) { visibleChangeTrigger(); }
void ExplorerRL::objectPropertyChanged(const SetObjectProperty *) { visibleChangeTrigger(); }
void ExplorerRL::tileRevealed(const FowTilesType &) { visibleChangeTrigger(); }
void ExplorerRL::tileHidden(const FowTilesType &) { visibleChangeTrigger(); }
void ExplorerRL::heroCreated(const CGHeroInstance *) { visibleChangeTrigger(); }
void ExplorerRL::heroInGarrisonChange(const CGTownInstance *) { visibleChangeTrigger(); }
void ExplorerRL::battleResultsApplied() { visibleChangeTrigger(); }
void ExplorerRL::playerStartsTurn(PlayerColor) { visibleChangeTrigger(); }
void ExplorerRL::invalidatePaths() { visibleChangeTrigger(); }
void ExplorerRL::receivedResource() { visibleChangeTrigger(); }
void ExplorerRL::heroMovePointsChanged(const CGHeroInstance *) { visibleChangeTrigger(); }
void ExplorerRL::garrisonsChanged(ObjectInstanceID, ObjectInstanceID) { visibleChangeTrigger(); }

std::optional<BattleAction> ExplorerRL::makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &)
{
	// Auto-battle is an explicitly fixed part of this exploration experiment.
	// No adventure-layer tactical surrender policy is supplied.
	return std::nullopt;
}
