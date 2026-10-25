// Actual CServerHandler shutdown + actual AsyncRunner, without game rules/assets.
#include "../StdInc.h"
#include "../../client/CMT.h"
#include "../../client/Client.h"
#include "../../client/CServerHandler.h"
#include "../../client/GameInstance.h"
#include "../../client/mapView/mapHandler.h"
#include "../../lib/AsyncRunner.h"
#include "../../lib/battle/BattleAction.h"
#include "../../lib/callback/CAdventureAI.h"
#include <future>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
void requestHeadlessShutdown(int) {}
bool headlessShutdownRequested() { return true; }
[[noreturn]] void handleFatalError(const std::string & reason, bool) { throw std::runtime_error(reason); }
namespace
{
struct Probe final : INetworkTimerListener
{
	std::promise<void> tick;
	void onTimer() override { tick.set_value(); }
};
class Interface final : public CAdventureAI, public INetworkTimerListener
{
public:
	INetworkHandler & network;
	std::unique_ptr<AsyncRunner> runner;
	std::promise<void> scheduled, taskStarted, release;
	std::shared_future<void> released = release.get_future().share();
	bool finished = false;
	explicit Interface(INetworkHandler & value) : network(value) {}
	void initGameInterface(std::shared_ptr<Environment>, std::shared_ptr<CCallback>) override {}
	void yourTurn(QueryID) override {}
	void heroGotLevel(const CGHeroInstance *, PrimarySkill, std::vector<SecondarySkill> &, QueryID) override {}
	void commanderGotLevel(const CCommanderInstance *, std::vector<ui32>, QueryID) override {}
	void showBlockingDialog(const std::string &, const std::vector<Component> &, QueryID, int, bool, bool, bool) override {}
	void showGarrisonDialog(const CArmedInstance *, const CGHeroInstance *, bool, QueryID, const MetaString &) override {}
	void showTeleportDialog(const CGHeroInstance *, TeleportChannelID, TTeleportExitsList, bool, QueryID) override {}
	void showMapObjectSelectDialog(QueryID, const Component &, const MetaString &, const MetaString &, const std::vector<ObjectInstanceID> &) override {}
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID &, const BattleStateInfoForRetreat &) override { return std::nullopt; }
	std::string getBattleAIName() const override { return "shutdown fixture, no battle"; }
	void onTimer() override
	{
		runner = std::make_unique<AsyncRunner>();
		runner->run([this] { taskStarted.set_value(); released.wait(); });
		scheduled.set_value();
	}
	void finish() override
	{
		if(finished) return;
		Probe producerProbe;
		auto tick = producerProbe.tick.get_future();
		network.createTimer(producerProbe, 1ms);
		if(tick.wait_for(2s) != std::future_status::ready)
		{
			std::cerr << "AI drain occurred after producer stopped\n";
			std::_Exit(4);
		}
		release.set_value();
		runner->wait();
		runner.reset();
		finished = true;
	}
};
}
int main()
{
	try
	{
		for(int iteration = 0; iteration < 100; ++iteration)
		{
			GAME = std::make_unique<GameInstance>();
			GAME->setMapInstance(std::make_unique<CMapHandler>(nullptr));
			auto & server = GAME->server();
			server.client = std::make_unique<CClient>();
			auto interface = std::make_shared<Interface>(server.getNetworkHandler());
			server.client->playerint.emplace(PlayerColor(0), interface);
			auto scheduled = interface->scheduled.get_future();
			auto started = interface->taskStarted.get_future();
			server.getNetworkHandler().createTimer(*interface, 1ms);
			if(scheduled.wait_for(2s) != std::future_status::ready || started.wait_for(2s) != std::future_status::ready)
				throw std::runtime_error("Fixture task did not start");
			server.endNetwork(); // exact production shutdown, no duplicate implementation
			if(!interface->finished)
			{
				std::cerr << "Production shutdown joined producer before AI drain\n";
				std::_Exit(3); // avoid deliberately completing a dangling TBB vertex
			}
			server.client.reset();
			GAME.reset();
		}
		std::cout << "100 actual headless shutdown order checks passed; no gameplay simulated\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
