// Real evidence/guard implementation: cancellation cannot append or send.
#include "StdInc.h"
#include "ContextualEvidence.h"
#include "lib/networkPacks/PacksForClient.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

void check(bool condition) { if(!condition) throw std::runtime_error("terminal cancellation check failed"); }
std::string contents(const std::string & path)
{
	std::ifstream stream(path);
	return std::string(std::istreambuf_iterator<char>(stream), {});
}
int main(int argc, char ** argv)
{
	try
	{
		check(argc == 3);
		const bool builtin = std::string(argv[1]) == "builtin";
		setenv("VCMI_EXPERIMENT_DIR", argv[2], 1);
		setenv("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1", 1);
		setenv("EXPLORER_RL_INTERFACE", "adventure-dialogs-v1", 1);
		if(builtin)
		{
			setenv("EXPLORER_RL_BUILTIN_OPPONENT", "Nullkiller", 1);
			setenv("EXPLORER_RL_PLAYERS", "0", 1);
		}
		auto & evidence = explorer_rl::ContextualEvidence::live("cancel-test", true, true, true, true, true);
		// Simulate pre-event scheduling before a concurrent terminal packet.
		evidence.beforeEvent("client", "request");
		{
			ExperimentEventGuard transaction("client", "packet");
			ExperimentTerminalBarrier barrier;
			barrier.episode = "cancel-test";
			barrier.interface = "adventure-dialogs-v1";
			barrier.barrierId = 1;
			evidence.terminalBarrierComplete(barrier);
		}
		const std::string path = std::string(argv[2]) + "/contextual-replay.jsonl";
		const auto sealed = contents(path);
		check(explorer_rl::ContextualEvidence::cancelled());
		int typed = 0, strict = 0, enteredBodies = 0;
		try { ExperimentEventGuard request("client", "request"); ++enteredBodies; }
		catch(const ExperimentTerminalFinished &) { ++typed; }
		catch(const std::runtime_error &) { ++strict; }
		// Recheck after acquiring the event mutex closes the before/enter race.
		try { evidence.eventEntered("client", "request"); ++enteredBodies; }
		catch(const ExperimentTerminalFinished &) { ++typed; }
		catch(const std::runtime_error &) { ++strict; }
		check(typed == (builtin ? 2 : 0) && strict == (builtin ? 0 : 2));
		check(enteredBodies == 0 && experimentEventActor().empty());
		try { evidence.beforeEvent("player0", "decision"); }
		catch(const std::runtime_error &) { ++strict; }
		try { evidence.eventEntered("client", "packet"); }
		catch(const std::runtime_error &) { ++strict; }
		try { evidence.message(0, JsonNode()); }
		catch(const std::runtime_error &) { ++strict; }
		check(strict == (builtin ? 3 : 5));
		check(contents(path) == sealed && !experimentHasFailed());
		std::cout << "terminal cancellation: " << argv[1] << " passed\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
