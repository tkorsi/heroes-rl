/*
 * ReplayWorker.cpp, part of the Heroes RL experiment tools for VCMI
 * License: GNU General Public License v2.0 or later
 */
#include "StdInc.h"
#include "ReplayWorker.h"
#include "ForkEpisode.h"
#include "WorkerProtocol.h"
#include "../lib/GameLibrary.h"
#include "../lib/CConsoleHandler.h"
#include "../lib/VCMIDirs.h"
#include "../lib/json/JsonNode.h"
#include "../lib/logging/CBasicLogConfigurator.h"

int replayWorker(int argc, const char * argv[], int (*run)(int, const char **))
{
#ifndef __linux__
	std::cerr << "Experimental replay worker requires Linux\n";
	return 2;
#else
	try
	{
		if(argc != 3 || std::string(argv[1]) != "--experimental-worker")
			throw std::runtime_error("Usage: vcmireplay --experimental-worker <positive job limit milliseconds>");
		size_t consumed = 0;
		const std::string limitText = argv[2];
		const auto limit = std::stoll(limitText, &consumed);
		if(consumed != limitText.size() || limit <= 0 || limit > 3600000)
			throw std::runtime_error("Job limit must be 1..3600000 milliseconds");
		optimizer::requireSingleThread();
		// stdout is protocol-only, including while VCMI writes startup logs.
		const int protocol = dup(STDOUT_FILENO);
		if(protocol < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
			throw std::runtime_error("Cannot isolate worker protocol");
		return optimizer::serveWorker(STDIN_FILENO, protocol, std::chrono::milliseconds(limit), []()
		{
			CConsoleHandler console;
			CBasicLogConfigurator logging(VCMIDirs::get().userLogsPath() / "VCMI_Worker_log.txt", &console);
			logging.configureDefault();
			LIBRARY = new GameLibrary;
			LIBRARY->initializeFilesystem(false);
			logging.configure();
			LIBRARY->initializeLibrary();
			logging.deconfigure();
		}, [&](const std::vector<std::string> & args)
		{
			std::vector<const char *> pointers{argv[0]};
			for(const auto & argument : args) pointers.push_back(argument.c_str());
			return run(static_cast<int>(pointers.size()), pointers.data());
		});
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 3;
	}
#endif
}
