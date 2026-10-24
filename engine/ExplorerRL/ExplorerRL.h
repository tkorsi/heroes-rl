#pragma once

#include "../../lib/callback/CAdventureAI.h"
#include "../../lib/json/JsonNode.h"
#include "ContextualState.h"
#include "NoticeParser.h"
#include "VisibleState.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
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

	// Windows and notifications the movement interface ignores, as the default
	// receivers do. The contextual interface records audited notices and stops
	// at everything else instead of silently swallowing it.
	void showInfoNotice(EInfoWindowMode, const MetaString &, const std::vector<Component> &) override;
	void showInfoDialog(EInfoWindowMode, const std::string &, const std::vector<Component> &, int) override;
	void showShipyardDialog(const IShipyard *) override;
	void showPuzzleMap() override;
	void viewWorldMap() override;
	void showHillFortWindow(const CGObjectInstance *, const CGHeroInstance *) override;
	void showThievesGuildWindow(const CGObjectInstance *) override;
	void showQuestLog() override;
	void showWorldViewEx(const std::vector<ObjectPosInfo> &, bool) override;

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
	int64_t submittedActionID = -1; ///< the last policy action issued to the server, protected by stateMutex
	int64_t decisionID = 0;
	int64_t completedBattles = 0;
	explorer_rl::VisibleTracker visibleTracker; ///< protected by stateMutex
	int64_t visibleSequence = 0; ///< protected by stateMutex
	std::optional<BattleSide> battleSide; ///< own side in the current battle, protected by stateMutex
	std::optional<bool> battleWon; ///< protected by stateMutex
	bool profile = false;
	int64_t visibleCaptureMicros = 0; ///< since the last profiled message, protected by stateMutex
	int64_t visibleCaptures = 0; ///< protected by stateMutex
	/// Engine-side replay evidence (VCMI_EXPERIMENT_REPLAY=1): the pack position
	/// and digest of each reported capture and of each policy message. It names
	/// engine IDs and counts opponent packs, so it never reaches a policy.
	bool ledgerEnabled = false;
	std::FILE * observationLedger = nullptr; ///< protected by stateMutex
	struct Capture
	{
		uint64_t pack;
		uint16_t packType;
		uint64_t digest;
	};
	std::optional<Capture> lastCapture; ///< protected by stateMutex
	int64_t ledgerMessages = 0; ///< protected by stateMutex
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
	/// Records the capture a message's events end at; requires the game-state read view.
	void noteLedgerMessage(const char * type);
	void visibleChangeTrigger() noexcept;
	void enqueueDecision(QueryID query, JsonNode payload);
	void handleDecision(Decision decision);
	/// Issues one request and waits for its acknowledgement. A policy action passes its ID: it is issued, and
	/// becomes the terminal's consumption watermark, only while this player's game is in progress; otherwise
	/// nothing is issued and `issued` stays false.
	bool executeRequest(const std::function<void()> & request, bool waitForAdventure = true, int64_t policyActionID = -1,
		bool * issued = nullptr);
	bool ended();
	void unsupported(const std::string & operation);

	// -- adventure-windows-v1 (EXPLORER_RL_INTERFACE), see ExplorerRLContext.cpp --
	bool contextual = false;
	/// An open hero or town window. Its fields were read when it opened; no
	/// game command can run while it is open, so they cannot go stale.
	struct OpenWindow
	{
		bool town = false;
		ObjectInstanceID subject;
	};
	std::optional<OpenWindow> window; ///< worker only
	int64_t revision = 0; ///< decisions and context changes, worker only
	explorer_rl::ContextTracker contextTracker; ///< protected by stateMutex
	/// The in-flight contextual action. gameOver reports its result when the
	/// game ends before the worker could, which requires every issued command
	/// to be acknowledged. Protected by stateMutex.
	struct ContextAction
	{
		int64_t decision = -1;
		bool endTurn = false;
		ObjectInstanceID hero;
		int3 destination;
		int commands = 0; ///< issued native commands
		int accepted = 0; ///< acknowledged and accepted native commands
		bool pending = false; ///< the last issued command awaits its acknowledgement
		bool lastAccepted = false;
		bool rejected = false;
		bool deviated = false; ///< the hero was not where the plan required
		bool finished = false; ///< every planned step was walked
		int64_t causesAtIssue = 0; ///< stopCauses when the last command was issued
	};
	std::optional<ContextAction> action;
	int issuingCommand = -1; ///< command index being sent, for requestSent; protected by stateMutex
	std::unordered_map<int, int> commandRequests; ///< request ID -> command index of the in-flight action; protected by stateMutex
	int64_t stopCauses = 0; ///< visits, battle events, notices and hero losses so far; protected by stateMutex
	std::optional<explorer_rl::NoticeParse> pendingNotice; ///< network thread only

	enum class Command
	{
		ENDED, ///< the game ended or the run stopped, before or after the command left
		DECLINED, ///< the precondition failed and nothing was issued
		REJECTED,
		ACCEPTED
	};

	JsonNode contextualReady();
	void runContextualTurn(QueryID queryID);
	void noteContextChanges(); ///< requires the stable state, like noteVisibleChanges
	/// Takes the ordered events, turning acknowledgement markers of the in-flight
	/// action's commands into acknowledged events and dropping all others. The
	/// action's result leaves with them, so the action is cleared.
	JsonNode takeContextEvents();
	JsonNode contextSnapshotJson(const explorer_rl::ContextSnapshot & snapshot) const;
	JsonNode contextResult(const ContextAction & current); ///< requires the stable state and stateMutex
	JsonNode heroWindow(const CGHeroInstance & hero) const;
	JsonNode townWindow(const CGTownInstance & town) const;
	/// Requires the stable state and an owned hero. Returns the result of a move
	/// that needs no native command, consumed by the adapter; otherwise starts
	/// the in-flight action, consumed by the engine, with the steps to walk.
	std::optional<JsonNode> planMove(const CGHeroInstance & hero, int64_t decision, const int3 & destination, std::vector<int3> & steps);
	/// Issues one native command of the in-flight action when the precondition
	/// holds under the same stable state, then waits for its acknowledgement and
	/// for movement and battles to finish.
	Command issueCommand(const std::function<void()> & request, const std::function<bool()> & precondition = {});
	/// Records that a command's request and its acknowledgement are both known; requires stateMutex.
	void settleCommand(int command, bool accepted);
	void pushEvent(JsonNode event, bool stopCause = false);
};
