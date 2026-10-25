// Actual shared-library evidence hooks; no game, map, model or mock engine.
#include "StdInc.h"
#include "lib/callback/ExperimentHooks.h"
#include "lib/callback/CGameInterface.h"
#include "lib/networkPacks/PacksForClient.h"
#include "lib/networkPacks/PacksForServer.h"
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

namespace
{
int checks = 0;
void check(bool value)
{
	++checks;
	if(!value)
		throw std::runtime_error("native terminal barrier check failed");
}
struct Hooks final : ExperimentEvidenceHooks
{
	std::vector<std::string> events;
	std::promise<void> waiting;
	std::atomic<bool> contenderEntered{false};
	bool failOnLeave = false;
	void beforeEvent(const std::string & actor, const std::string & operation) override
	{
		if(actor == "contender")
			waiting.set_value();
	}
	void eventEntered(const std::string & actor, const std::string & operation) override
	{
		events.push_back("enter:" + actor + ":" + operation);
		if(actor == "contender")
			contenderEntered = true;
	}
	void eventLeaving(const std::string & actor, const std::string & operation) override
	{
		events.push_back("leave:" + actor + ":" + operation);
		if(failOnLeave)
			throw std::runtime_error("deliberate diagnostic flush failure");
	}
};
}

int main(int argc, char ** argv)
{
	try
	{
		if(argc != 2)
			throw std::runtime_error("requires a new temporary output directory");
		EndTurn turn;
		turn.player = PlayerColor(0);
		check(supportedExperimentTerminalRequest(turn, false));
		turn.player = PlayerColor(1);
		check(supportedExperimentTerminalRequest(turn, false));
		turn.player = PlayerColor::NEUTRAL;
		check(!supportedExperimentTerminalRequest(turn, true));
		MakeAction battle;
		battle.player = PlayerColor::NEUTRAL;
		check(!supportedExperimentTerminalRequest(battle, false));
		check(supportedExperimentTerminalRequest(battle, true));
		battle.player = PlayerColor::UNFLAGGABLE;
		check(!supportedExperimentTerminalRequest(battle, true));
		battle.player = PlayerColor(2);
		check(!supportedExperimentTerminalRequest(battle, true));
		battle.player = PlayerColor(0);
		check(supportedExperimentTerminalRequest(battle, false));
		auto hooks = std::make_shared<Hooks>();
		setExperimentEvidenceHooks(hooks);
		unsetenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE");
		{
			ExperimentEventGuard disabled("legacy", "read");
			check(!contextualEvidenceEnabled());
		}
		check(hooks->events.empty());
		setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
		std::thread contender;
		{
			std::string actor = "player0";
			ExperimentEventGuard outer(actor.c_str(), "read");
			actor = "mutated-after-construction";
			{
				ExperimentEventGuard nested("ignored", "send");
				check(experimentEventActor() == "player0");
				check(hooks->events == std::vector<std::string>{"enter:player0:read"});
			}
			contender = std::thread([] { ExperimentEventGuard guard("contender", "read"); });
			hooks->waiting.get_future().get();
			// beforeEvent ran, but the second operation cannot enter this transaction.
			check(!hooks->contenderEntered.load());
		}
		contender.join();
		check(hooks->events == std::vector<std::string>{"enter:player0:read", "leave:player0:read", "enter:contender:read", "leave:contender:read"});
		check(experimentEventActor().empty());
		{
			ExperimentEventGuard packet("client", "packet");
			beginExperimentPacket(265);
			auto position = experimentReceiptPosition();
			check(position.index == 1 && position.completed == 0 && position.type == 265 && position.phase == ExperimentPacketPhase::Before);
			noteExperimentPackApplied(265);
			position = experimentReceiptPosition();
			check(position.phase == ExperimentPacketPhase::Applied && position.completed == 0);
			noteExperimentPacketCallbacksComplete();
			position = experimentReceiptPosition();
			check(position.phase == ExperimentPacketPhase::CallbacksComplete && position.completed == 1);
			bool duplicateRejected = false;
			try { noteExperimentPacketCallbacksComplete(); }
			catch(const std::runtime_error &) { duplicateRejected = true; }
			check(duplicateRejected);
			check(experimentPackPosition().index == 1 && experimentPackPosition().type == 265);
			setenv("VCMI_EXPERIMENT_DIR", argv[1], 1);
			ExperimentTerminalBarrier barrier;
			barrier.episode = "synthetic-hook-check";
			barrier.interface = "adventure-town-v1";
			barrier.requestPlayer = PlayerColor::NEUTRAL;
			barrier.requestID = 7;
			barrier.requestType = 198; // pinned MakeAction registration
			barrier.clientPackets = 1;
			barrier.redWon = true;
			writeExperimentTerminalBarrier(barrier, false);
			writeExperimentTerminalBarrier(barrier, true);
			bool overwriteRejected = false;
			try { writeExperimentTerminalBarrier(barrier, true); }
			catch(const std::runtime_error &) { overwriteRejected = true; }
			check(overwriteRejected);
		}
		bool failureReported = false;
		setExperimentFailureHandler([&](const std::string & reason) { failureReported = reason.find("deliberate diagnostic flush failure") != std::string::npos; });
		hooks->failOnLeave = true;
		{
			ExperimentEventGuard failed("client", "packet");
			check(!experimentHasFailed());
		}
		check(failureReported && experimentHasFailed());
		setExperimentFailureHandler({});
		setExperimentEvidenceHooks({});
		std::cout << "native terminal barrier hooks: " << checks << " checks passed\n";
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
