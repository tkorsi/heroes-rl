/*
 * ReplayWorker.cpp, part of the Heroes RL experiment tools for VCMI
 * License: GNU General Public License v2.0 or later
 */
#include "StdInc.h"
#include "ReplayWorker.h"
#include "ForkEpisode.h"
#include "../lib/GameLibrary.h"
#include "../lib/CConsoleHandler.h"
#include "../lib/VCMIDirs.h"
#include "../lib/json/JsonNode.h"
#include "../lib/logging/CBasicLogConfigurator.h"

#ifdef __linux__
namespace
{
bool readJob(std::string & line)
{
	line.clear();
	char byte;
	while(std::cin.get(byte))
	{
		if(byte == '\n')
			return true;
		if(line.size() == 65536)
			throw std::runtime_error("Job exceeds 65536 bytes");
		line += byte;
	}
	if(!std::cin.eof())
		throw std::runtime_error("Cannot read worker input");
	return !line.empty();
}
}
#endif

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
		FILE * replies = fdopen(protocol, "w");
		if(!replies)
			throw std::runtime_error("Cannot open worker protocol");
		const auto start = std::chrono::steady_clock::now();
		{
			CConsoleHandler console;
			CBasicLogConfigurator logging(VCMIDirs::get().userLogsPath() / "VCMI_Worker_log.txt", &console);
			logging.configureDefault();
			LIBRARY = new GameLibrary;
			LIBRARY->initializeFilesystem(false);
			logging.configure();
			LIBRARY->initializeLibrary();
			logging.deconfigure();
		}
		optimizer::requireSingleThread();
		fprintf(replies, "{\"kind\":\"ready\",\"schema\":\"heroes-replay-worker-v1\",\"load_seconds\":%.9f}\n",
			std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
		fflush(replies);
		std::string line;
		while(readJob(line))
		{
			if(line.empty() || line.size() > 65536)
				throw std::runtime_error("Expected a JSON argument array of at most 65536 bytes");
			JsonParsingSettings parsing;
			parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
			parsing.strict = true;
			JsonNode job(reinterpret_cast<const std::byte *>(line.data()), line.size(), parsing, "worker job");
			if(!job.isVector())
				throw std::runtime_error("Expected a JSON argument array");
			std::vector<std::string> arguments{argv[0]};
			std::set<std::string> seen;
			const std::set<std::string> values{"--log", "--report", "--observations", "--trace", "--contextual"};
			for(size_t i = 0; i < job.Vector().size(); ++i)
			{
				const auto & option = job[i];
				if(option.getType() != JsonNode::JsonType::DATA_STRING || !seen.insert(option.String()).second)
					throw std::runtime_error("Job options must be unique strings");
				const auto & name = option.String();
				arguments.push_back(name);
				if(values.count(name))
				{
					if(++i >= job.Vector().size() || job[i].getType() != JsonNode::JsonType::DATA_STRING
						|| job[i].String().empty() || job[i].String().find('\0') != std::string::npos)
						throw std::runtime_error("Missing job option value");
					arguments.push_back(job[i].String());
					if((name == "--report" || name == "--trace") && boost::filesystem::exists(arguments.back()))
						throw std::runtime_error("Worker output already exists");
				}
				else if(name != "--prefix")
					throw std::runtime_error("Unsupported worker option: " + name);
			}
			if(!seen.count("--log") || !seen.count("--report"))
				throw std::runtime_error("Every job requires --log and --report");
			std::vector<const char *> pointers;
			for(const auto & argument : arguments)
				pointers.push_back(argument.c_str());
			const auto result = optimizer::forkEpisode([&]()
			{
				fclose(replies);
				return run(static_cast<int>(pointers.size()), pointers.data());
			}, std::chrono::milliseconds(limit));
			fprintf(replies, "{\"kind\":\"completed\",\"exit_code\":%d,\"signal\":%d,\"timed_out\":%s,\"seconds\":%.9f}\n",
				result.exitCode, result.signal, result.timedOut ? "true" : "false", result.seconds);
			fflush(replies);
		}
		fclose(replies);
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 3;
	}
#endif
}
