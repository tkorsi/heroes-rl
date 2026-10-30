#pragma once

#include "../../lib/callback/CAdventureAI.h"
#include "../../lib/json/JsonNode.h"
#include "VisibleState.h"
#include "Contextual.h"
#include "../../nativesession/PolicyTransport.h"
#include "../../lib/callback/ExperimentHooks.h"

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
	void showStructuredBlockingDialog(const MetaString &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override;
	void showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override;
	void showStructuredInfoDialog(EInfoWindowMode, const MetaString &, const std::vector<Component> &, int) override;
	void showInfoDialog(EInfoWindowMode, const std::string &, const std::vector<Component> &, int) override;
	void showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) override;
	void showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) override;
	void showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) override;
	void showRecruitmentDialog(const CGDwelling *, const CArmedInstance *, int, QueryID) override;
	void showShipyardDialog(const IShipyard *) override;
	void showPuzzleMap() override;
	void viewWorldMap() override;
	void showWorldViewEx(const std::vector<ObjectPosInfo> &, bool) override;
	void showHillFortWindow(const CGObjectInstance *, const CGHeroInstance *) override;
	void showThievesGuildWindow(const CGObjectInstance *) override;
	void showQuestLog() override;
	void askToAssembleArtifact(const ArtifactLocation &) override;
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
	// Atomic shared_ptr free functions also work on libc++ without atomic<shared_ptr>.
	std::shared_ptr<native_session::PolicyTransport> policyTransport;
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
	int timeoutSeconds = 30;
	int snapshotEvery = 1; ///< observations carry the visible tile list every N decisions, never when 0
	int64_t observations = 0;
	int64_t actionID = 0;
	int64_t submittedActionID = -1; ///< the last policy action issued to the server, protected by stateMutex
	int64_t decisionID = 0;
	int64_t completedBattles = 0;
	bool contextual = false;
	bool expandedInterface = false; ///< opt-in protocol-10 native adventure capabilities
	bool ownArmyInformation = false; ///< opt-in v8 passive owned-army composition and popup statistics
	std::string evidenceActor;
	bool townEconomy = false; ///< separate protocol-5 profile; never changes v2
	bool heroRecruitment = false; ///< opt-in protocol-7 owned-town tavern and multiple heroes
	bool armyTransfer = false; ///< opt-in protocol-6 overview armies and exact transfers
	bool adventureDialogs = false; ///< opt-in protocol-9 explicit native adventure choices
	bool armyManagement = false; ///< opt-in protocol-8 explicit army operations and exchange queries
	bool contextualIssueLockHeld = false; ///< worker thread only, during request()/requestSent()
	explorer_rl::ContextualState contextState; ///< protected by stateMutex
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
		ExperimentEventGuard event;
		std::shared_lock<std::shared_mutex> lock;

	public:
		explicit GameStateRead(const char * actor);
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
		bool * issued = nullptr, const std::function<bool()> & beforeIssue = {});
	bool ended();
	void unsupported(const std::string & operation);
	void runContextualTurn(QueryID queryID, bool modalOnly = false);
	void captureContextual(); ///< stable game state; acquires stateMutex
	void captureContextualLocked(); ///< stable game state and stateMutex held
	JsonNode contextualObservation(); ///< stable game state and stateMutex held
	JsonNode takeContextualRecords(); ///< acquires stateMutex
	void contextualEvent(JsonNode event); ///< stable game state; acquires stateMutex
	void contextualTerminal(const char * outcome); ///< network thread, under sendMutex
	bool executeContextualAction(const JsonNode & action, int64_t decision);
	bool executeTownAction(const JsonNode & action, int64_t decision);
	bool executeRecruitmentAction(const JsonNode & action, int64_t decision);
	JsonNode tavernPanel(const CGTownInstance & town); ///< only the explicitly opened v5 tavern
	bool executeTavernAction(const JsonNode & action, int64_t decision);
	JsonNode townArmyExchange(const CGTownInstance & town); ///< only the explicit v4 overview
	bool executeTransferAction(const JsonNode & action, int64_t decision);
	JsonNode contextualArmyViews(); ///< only the current explicitly displayed owned rows
	JsonNode stackPanel(); ///< only the opened stack or selected upgrade confirmation
	bool executeArmyAction(const JsonNode & action, int64_t decision);
	bool executeStackAction(const JsonNode & action, int64_t decision);
	bool executeGarrisonAction(const JsonNode & action, int64_t decision);
	bool activeOwnHeroExchange(int firstEngineID, int secondEngineID) const; ///< stateMutex held
	void installOwnHeroExchange(); ///< stable game state and stateMutex held, no pending action
	bool executeExchangeClose(const JsonNode & action, int64_t decision);
	void installAdventureDialog(); ///< stable game state and stateMutex held, parent action finished
	JsonNode adventureDialogPanel() const; ///< immutable received payload only, stateMutex held
	bool executeDialogAction(const JsonNode & action, int64_t decision);
	JsonNode capabilityControls(); ///< only controls in the explicitly opened parent
	JsonNode capabilityPanel(); ///< player-visible native contents and legal typed controls
	bool executeCapabilityAction(const JsonNode & action, int64_t decision);
	void installCapabilityDialog(); ///< state/read locks held
	void receiveLevelUp(const CGHeroInstance *, PrimarySkill, const std::vector<SecondarySkill> &, QueryID);
	void receiveGarrison(const CArmedInstance *, const CGHeroInstance *, bool, QueryID);
	void contextualResult(JsonNode result, int64_t decision);
};
