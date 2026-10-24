#include "StdInc.h"
#include "ExplorerRL.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/gameState/CGameState.h"
#include "../../lib/gameState/EVictoryLossCheckResult.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapping/TerrainTile.h"
#include "../../lib/networkPacks/PacksForClient.h"

#include <arpa/inet.h>
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
	const int port = environmentInteger("EXPLORER_RL_PORT", 0, 1, 65535);
	if(port == 0)
		throw std::runtime_error("EXPLORER_RL_PORT is required");
	timeoutSeconds = environmentInteger("EXPLORER_RL_TIMEOUT", 30, 1, 300);
	const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
	if(fd < 0)
		throw std::runtime_error("Cannot create policy socket");
	socketFD.store(fd);
#ifdef SO_NOSIGPIPE
	const int yes = 1;
	::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
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
	result["player"].Integer() = playerID.getNum();
	result["episode"].String() = episode;
	return result;
}

void ExplorerRL::sendMessage(const JsonNode & message)
{
	// VCMI's compact writer can retain formatting newlines. JSON strings escape
	// their own newlines, so removing literal CR/LF produces a JSONL record.
	std::string line = message.toCompactString();
	line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
	line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
	line += '\n';
	std::lock_guard lock(sendMutex);
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
}

JsonNode ExplorerRL::receiveMessage()
{
	while(true)
	{
		const auto newline = receiveBuffer.find('\n');
		if(newline != std::string::npos)
		{
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
		while(true)
		{
			QueryID query;
			{
				std::unique_lock lock(stateMutex);
				changed.wait(lock, [this] { return stopping || terminal || !failure.empty() || !turns.empty(); });
				if(stopping || terminal)
					return;
				if(!failure.empty())
					throw std::runtime_error(failure);
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
			failure = error.what();
		}
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
		// BattleAI uses the same callback. Retain only our one outstanding request.
		// Responses may arrive before requestSent returns on the sending thread.
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

void ExplorerRL::battleEnded()
{
	{
		std::lock_guard lock(stateMutex);
		inBattle = false;
	}
	changed.notify_all();
}

bool ExplorerRL::executeRequest(const std::function<void()> & request)
{
	{
		std::lock_guard lock(stateMutex);
		pendingRequest = -1;
		acknowledgements.clear();
	}
	{
		std::shared_lock gameLock(CGameState::mutex);
		request();
	}
	std::unique_lock lock(stateMutex);
	const bool completed = changed.wait_for(lock, std::chrono::seconds(timeoutSeconds), [this] {
		return stopping || terminal || !failure.empty()
			|| (pendingRequest >= 0 && acknowledgements.contains(pendingRequest) && !moving && !inBattle);
	});
	if(!completed)
		throw std::runtime_error("Engine request or battle completion timed out");
	if(!failure.empty())
		throw std::runtime_error(failure);
	if(stopping || terminal)
		return false;
	return acknowledgements.at(pendingRequest);
}

JsonNode ExplorerRL::observation(std::array<bool, 9> & mask, ObjectInstanceID & heroID, int3 & position)
{
	std::shared_lock gameLock(CGameState::mutex);
	const auto heroes = callback->getHeroesInfo();
	if(heroes.size() != 1 || heroes.front()->getOwner() != playerID)
		throw std::runtime_error("ExplorerRL scenario requires exactly one owned hero");
	const auto * hero = heroes.front();
	heroID = hero->id;
	position = hero->visitablePos();
	const auto size = callback->getMapSize();
	auto result = message("observation");
	result["action_id"].Integer() = actionID;
	result["day"].Integer() = callback->getDate(Date::DAY);
	result["hero_id"].Integer() = heroID.getNum();
	result["x"].Integer() = position.x;
	result["y"].Integer() = position.y;
	result["z"].Integer() = position.z;
	result["width"].Integer() = size.x;
	result["height"].Integer() = size.y;
	result["levels"].Integer() = size.z;
	result["movement"].Integer() = hero->movementPointsRemaining();
	result["movement_max"].Integer() = hero->movementPointsLimit();
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
				item["terrain"].Integer() = tile->getTerrainID().getNum();
				item["blocked"].Bool() = tile->blocked();
				item["visitable"].Bool() = tile->visitable();
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
	const auto * source = callback->getTile(position, false);
	for(size_t i = 0; i < DIRECTIONS.size(); ++i)
	{
		const int3 destination(position.x + DIRECTIONS[i][0], position.y + DIRECTIONS[i][1], position.z);
		if(hero->movementPointsRemaining() <= 0 || !callback->isInTheMap(destination))
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
		result["legal_actions"].Vector().emplace_back(allowed);
	return result;
}

void ExplorerRL::runTurn(QueryID queryID)
{
	if(!executeRequest([this, queryID] { callback->selectionMade(0, queryID); }))
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
		sendMessage(observation(mask, heroID, position));
		const auto response = receiveMessage();
		if(ended())
			return;
		if(!response["action_id"].isNumber() || response["action_id"].Integer() != actionID || !response["action"].isNumber())
			throw std::runtime_error("Policy response must contain matching action_id and integer action");
		const auto action = response["action"].Integer();
		if(action < 0 || action >= static_cast<int64_t>(mask.size()) || !mask[action])
			throw std::runtime_error("Policy selected a masked or unsupported action");
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
		auto result = message("action_result");
		result["action_id"].Integer() = actionID++;
		result["action"].Integer() = action;
		result["accepted"].Bool() = accepted;
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
		auto record = message("terminal");
		record["outcome"].String() = result.victory() ? "win" : "loss";
		sendMessage(record);
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

void ExplorerRL::heroGotLevel(const CGHeroInstance *, PrimarySkill, std::vector<SecondarySkill> &, QueryID) { unsupported("hero level-up"); }
void ExplorerRL::commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) { unsupported("commander level-up"); }
void ExplorerRL::showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) { unsupported("blocking dialog"); }
void ExplorerRL::showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) { unsupported("garrison exchange"); }
void ExplorerRL::showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) { unsupported("teleport"); }
void ExplorerRL::showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) { unsupported("object selection"); }

std::optional<BattleAction> ExplorerRL::makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &)
{
	// Auto-battle is an explicitly fixed part of this exploration experiment.
	// No adventure-layer tactical surrender policy is supplied.
	return std::nullopt;
}
