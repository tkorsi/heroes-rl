#pragma once

#include "../../lib/callback/CAdventureAI.h"
#include "../../lib/json/JsonNode.h"
#include "VisibleState.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

/// A restricted adventure interface, not an adventure strategy.
/// The external learner selects every movement attempt and end-turn action.
class ExplorerRL final : public CAdventureAI
{
public:
	~ExplorerRL() override;
	void initGameInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CCallback> CB) override;
	std::string getBattleAIName() const override { return "BattleAI"; }
	void yourTurn(QueryID queryID) override;
	void finish() override;
	void requestSent(const CPackForServer * pack, int requestID) override;
	void requestRealized(PackageApplied * pack) override;
	void playerBlocked(int reason, bool start) override;
	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void battleEnd(const BattleID & battleID, const BattleResult * result, QueryID queryID) override;
	void battleEnded() override;
	void heroVisit(const CGHeroInstance *, const CGObjectInstance *, bool) override;
	void gameOver(PlayerColor player, const EVictoryLossCheckResult & result) override;
	void heroGotLevel(const CGHeroInstance *, PrimarySkill, std::vector<SecondarySkill> &, QueryID) override;
	void commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) override;
	void showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override;
	void showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) override;
	void showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) override;
	void showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) override;
	void showRecruitmentDialog(const CGDwelling *, const CArmedInstance *, int, QueryID) override;
	void showMarketWindow(const IMarket *, const CGHeroInstance *, QueryID) override;
	void showUniversityWindow(const IMarket *, const CGHeroInstance *, QueryID) override;
	void showTavernWindow(const CGObjectInstance *, const CGHeroInstance *, QueryID) override;
	void heroExchangeStarted(ObjectInstanceID, ObjectInstanceID, QueryID) override;
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &) override;

	/// Captures the player-visible and own state after every applied pack, so
	/// a transient sighting between two decisions is never skipped. Raw
	/// notifications such as heroMoved and objectRemoved stay unhandled and are
	/// never forwarded: a move can name a hidden endpoint and some removals
	/// are hidden.
	void packApplied() override;

private:
	std::shared_ptr<CCallback> callback;
	std::thread worker;
	std::atomic<int> socketFD{-1};
	std::mutex stateMutex;
	std::mutex sendMutex;
	std::condition_variable changed;
	struct Decision
	{
		QueryID query;
		JsonNode payload;
	};
	std::deque<QueryID> turns;
	std::deque<Decision> decisions;
	std::vector<JsonNode> events;
	std::unordered_map<int, bool> acknowledgements;
	bool stopping = false;
	bool terminal = false;
	bool moving = false;
	bool inBattle = false;
	int pendingRequest = -1;
	std::string failure;
	std::string episode;
	std::string receiveBuffer;
	int timeoutSeconds = 30;
	int snapshotEvery = 1; ///< observations carry the visible tile list every N decisions, never when 0
	int64_t observations = 0;
	int64_t actionID = 0;
	int64_t decisionID = 0;
	int64_t completedBattles = 0;
	explorer_rl::VisibleTracker visibleTracker; ///< protected by stateMutex
	int64_t visibleSequence = 0; ///< protected by stateMutex
	std::optional<BattleSide> battleSide; ///< own side in the current battle, protected by stateMutex
	std::optional<bool> battleWon; ///< protected by stateMutex
	bool profile = false;
	int64_t visibleCaptureMicros = 0; ///< since the last profiled message, protected by stateMutex
	int64_t visibleCaptures = 0; ///< protected by stateMutex
	int64_t lastEncodeMicros = 0; ///< protected by sendMutex
	int64_t lastSendMicros = 0; ///< protected by sendMutex
	int64_t lastMessageBytes = 0; ///< protected by sendMutex

	void run() noexcept;
	void runTurn(QueryID queryID);
	void connectPolicy();
	/// Builds and writes one message under sendMutex unless the episode has
	/// ended, and returns whether it did. See the definition for the ordering.
	bool buildAndSend(const std::function<JsonNode()> & build);
	void writeMessage(const JsonNode & message); ///< requires sendMutex
	JsonNode receiveMessage();
	JsonNode message(const std::string & type) const;
	/// A shared lock on the game state that refuses to nest on one thread:
	/// std::shared_mutex is not recursive, and a nested read queued behind a
	/// waiting writer would deadlock.
	class GameStateRead
	{
		std::shared_lock<std::shared_mutex> lock;

	public:
		GameStateRead();
		~GameStateRead();
		GameStateRead(const GameStateRead &) = delete;
		GameStateRead & operator=(const GameStateRead &) = delete;
	};

	/// A stable view for worker-thread reads. Network callbacks run on the only
	/// thread that applies packets and read without it.
	GameStateRead readGameState() const;
	JsonNode observation(std::array<bool, 9> & mask, ObjectInstanceID & heroID, int3 & position);
	JsonNode playerState();
	JsonNode takeEvents();
	JsonNode takeProfile(); ///< requires sendMutex
	void noteVisibleChanges();
	void visibleChangeTrigger() noexcept;
	void enqueueDecision(QueryID query, JsonNode payload);
	void handleDecision(Decision decision);
	bool executeRequest(const std::function<void()> & request, bool waitForAdventure = true);
	bool ended();
	void unsupported(const std::string & operation);
};
