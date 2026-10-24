#pragma once

#include "../../lib/callback/CAdventureAI.h"
#include "../../lib/json/JsonNode.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
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
	void battleEnded() override;
	void gameOver(PlayerColor player, const EVictoryLossCheckResult & result) override;
	void heroGotLevel(const CGHeroInstance *, PrimarySkill, std::vector<SecondarySkill> &, QueryID) override;
	void commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) override;
	void showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override;
	void showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) override;
	void showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) override;
	void showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) override;
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &) override;

private:
	std::shared_ptr<CCallback> callback;
	std::thread worker;
	std::atomic<int> socketFD{-1};
	std::mutex stateMutex;
	std::mutex sendMutex;
	std::condition_variable changed;
	std::deque<QueryID> turns;
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
	int64_t actionID = 0;

	void run() noexcept;
	void runTurn(QueryID queryID);
	void connectPolicy();
	void sendMessage(const JsonNode & message);
	JsonNode receiveMessage();
	JsonNode message(const std::string & type) const;
	JsonNode observation(std::array<bool, 9> & mask, ObjectInstanceID & heroID, int3 & position);
	bool executeRequest(const std::function<void()> & request);
	bool ended();
	void unsupported(const std::string & operation);
};
